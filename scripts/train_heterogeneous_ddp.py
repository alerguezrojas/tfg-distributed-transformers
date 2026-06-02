"""Entrenamiento DDP heterogéneo — GPU (verode21 V100) + CPU (verode16/18).

Reparte la carga proporcionalmente a la capacidad de cómputo de cada nodo:
  - rank 0 (GPU): procesa la fracción grande del dataset con batch_size alto
  - rank 1 (CPU): procesa la fracción pequeña con batch_size reducido

El backend gloo soporta tanto CUDA como CPU en el mismo grupo de procesos.

Lanzar (dos terminales tmux separadas):

    # Terminal 1 — verode21 (V100, rank 0):
    ssh verode21
    cd ~/tfg-distributed-transformers
    .venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \\
      --master_addr=verode21 --master_port=29500 \\
      scripts/train_heterogeneous_ddp.py \\
      --config configs/train_heterogeneous_ddp.yaml --trace simple

    # Terminal 2 — verode16 o verode18 (CPU, rank 1):
    ssh verode16
    cd ~/tfg-distributed-transformers
    .venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \\
      --master_addr=verode21 --master_port=29500 \\
      scripts/train_heterogeneous_ddp.py \\
      --config configs/train_heterogeneous_ddp.yaml --trace simple
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BigEarthNetDataset, get_transforms
from src.training.builder import TrainingSessionBuilder
from src.training.heterogeneous_sampler import HeterogeneousDistributedSampler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento DDP heterogéneo (GPU + CPU)"
    )
    parser.add_argument("--config", type=str,
                        default="configs/train_heterogeneous_ddp.yaml")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--model", type=str, help="Override nombre del modelo")
    parser.add_argument(
        "--trace", choices=["off", "simple", "deep"], default="simple",
    )
    parser.add_argument(
        "--layers", nargs="*",
        choices=["plot", "hooks", "confusion", "batch-monitor"],
        default=[],
    )
    parser.add_argument("--fn", nargs="*", choices=["timing", "energy"], default=[])
    parser.add_argument(
        "--metrics", nargs="*",
        choices=["loss", "f1", "accuracy", "precision_recall"],
        default=["loss", "f1", "accuracy", "precision_recall"],
    )
    parser.add_argument(
        "--batch-log-every", type=int, default=None, metavar="N",
        help="Log batch metrics cada N batches (default del config o 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dist_cfg = cfg.get("distributed", {})
    backend = dist_cfg.get("backend", "gloo")
    ranks_cfg = dist_cfg.get("ranks", [])

    # ── Init distributed ──────────────────────────────────────────────────────
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Dispositivo de este rank según la config
    if ranks_cfg and rank < len(ranks_cfg):
        rank_device_str = ranks_cfg[rank].get("device", "cpu")
    else:
        rank_device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if rank_device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    # Timestamp compartido desde rank 0 para coherencia en nombres de ficheros
    ts_list = [datetime.now().strftime("%d%m%Y_%H%M%S") if rank == 0 else ""]
    dist.broadcast_object_list(ts_list, src=0)
    timestamp = ts_list[0]

    # ── Config por rank ───────────────────────────────────────────────────────
    if ranks_cfg and rank < len(ranks_cfg):
        rank_batch_size = ranks_cfg[rank].get("batch_size", cfg["training"]["batch_size"])
        compute_weights = [r.get("compute_weight", 1) for r in ranks_cfg]
    else:
        rank_batch_size = cfg["training"]["batch_size"]
        compute_weights = [1] * world_size

    if rank == 0:
        total_bs = sum(
            (ranks_cfg[r].get("batch_size", cfg["training"]["batch_size"])
             if ranks_cfg and r < len(ranks_cfg) else cfg["training"]["batch_size"])
            for r in range(world_size)
        )
        print(f"DDP heterogéneo — {world_size} ranks | backend: {backend}")
        print(f"Batch global: {total_bs} | pesos: {compute_weights}")
        print(f"Rank 0 device: {device} | batch: {rank_batch_size}")

    # ── Dataset y DataLoaders ─────────────────────────────────────────────────
    train_ds = BigEarthNetDataset(
        cfg["data"]["root"], cfg["data"]["metadata"], split="train",
        transform=get_transforms("train"),
    )
    val_ds = BigEarthNetDataset(
        cfg["data"]["root"], cfg["data"]["metadata"], split="val",
        transform=get_transforms("val"),
    )

    train_sampler = HeterogeneousDistributedSampler(
        train_ds, weights=compute_weights, rank=rank, world_size=world_size,
        shuffle=True, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )

    num_workers = cfg["data"].get("num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=rank_batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=rank_batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Stack de entrenamiento via Builder ────────────────────────────────────
    # .with_heterogeneous_ddp() crea HeterogeneousDDPTrainer como base:
    #   - El modelo se envuelve con DDP una sola vez (dentro del trainer)
    #   - BatchMonitorDecorator registra sus hooks sobre HeterogeneousDDPTrainer
    #   - La normalización de gradientes por batch global es automática
    cfg_copy = {**cfg, "training": {**cfg["training"], "batch_size": rank_batch_size}}

    trace = args.trace if rank == 0 else "off"
    layers = (args.layers or []) if rank == 0 else []
    fn = args.fn or []
    metrics = args.metrics if args.metrics is not None else []

    builder = (
        TrainingSessionBuilder(
            cfg_copy, device, timestamp,
            rank=rank, world_size=world_size, distributed=True,
        )
        .with_heterogeneous_ddp(rank_batch_size)
        .with_output_mode("ddp_hetero")
        .with_trace(trace)
        .with_layers(*layers)
        .with_fn(*fn)
        .with_metrics(*metrics)
    )
    if args.model:
        builder = builder.with_model(args.model)
    if args.batch_log_every is not None:
        builder = builder.with_batch_log_every(args.batch_log_every)

    trainer = builder.build()

    if rank == 0:
        model_name = args.model or cfg["model"]["name"]
        print(f"Modelo: {model_name} | Trace: {trace} | Layers: {layers or 'ninguno'}")
        print(f"Muestras rank 0: {len(train_sampler)} train / {len(val_sampler)} val")

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    trainer.fit(train_loader, val_loader, epochs=epochs)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
