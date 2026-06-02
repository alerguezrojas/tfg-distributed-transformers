"""Distributed training script for BigEarthNet-S2 (multi-GPU, multi-node DDP).

Launch with torchrun:
    # Single node, 2 GPUs:
    torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/train_ddp_verode.yaml

    # Multi-node via Slurm (run from login node inside tmux):
    srun --partition=batch --nodes=2 --nodelist=verode16,verode21 \\
      --gres=gpu:tesla:1 --ntasks-per-node=1 --cpus-per-task=8 --time=48:00:00 \\
      bash -c 'cd ~/tfg-distributed-transformers && \\
        .venv/bin/torchrun \\
          --nnodes=2 --nproc_per_node=1 \\
          --node_rank=$SLURM_NODEID \\
          --master_addr=verode16 --master_port=29500 \\
          scripts/train_ddp.py --config configs/train_ddp_verode.yaml --trace simple'

    # Smoke test (1 GPU, validates DDP code path without a second GPU):
    torchrun --nproc_per_node=1 scripts/train_ddp.py \\
        --config configs/train_v3.yaml --model vit_tiny_patch16_224 --epochs 1

torchrun injects RANK, LOCAL_RANK, WORLD_SIZE as environment variables.
Only rank 0 prints to console and writes logs / checkpoints.

The batch_size in the config is PER GPU — the effective global batch size is
batch_size × world_size.  Adjust lr proportionally if needed (linear scaling rule).
"""

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


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed training — BigEarthNet-S2")
    parser.add_argument("--config", type=str, default="configs/train_cluster.yaml")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--model", type=str, help="Override model name (any timm ID)")
    parser.add_argument(
        "--trace", choices=["off", "simple", "deep"], default="simple",
        help="Logging controller mode (applies only to rank 0)",
    )
    parser.add_argument(
        "--layers", nargs="*",
        choices=["plot", "hooks", "confusion", "batch-monitor"],
        default=[],
        help="Stackable aspect decorators (applies only to rank 0)",
    )
    parser.add_argument(
        "--fn", nargs="*", choices=["timing", "energy"], default=[],
        help="Python @ decorators for train_epoch / eval_epoch",
    )
    parser.add_argument(
        "--metrics", nargs="*",
        choices=["loss", "f1", "accuracy", "precision_recall"],
        default=["loss", "f1", "accuracy", "precision_recall"],
        help="Metric reporters (only active for --trace off/simple on rank 0)",
    )
    parser.add_argument(
        "--batch-log-every", type=int, default=None, metavar="N",
        help="Log batch metrics every N batches (default 1). Applies only to rank 0.",
    )
    return parser.parse_args()


def main():
    # ── Distributed init ─────────────────────────────────────────────────────
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    backend = cfg.get("distributed", {}).get("backend", "nccl")
    use_cuda = backend == "nccl" and torch.cuda.is_available()

    local_rank = int(os.environ["LOCAL_RANK"])
    if use_cuda:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    # All ranks must use the same timestamp for consistent log/checkpoint naming
    ts_list = [datetime.now().strftime("%d%m%Y_%H%M%S") if rank == 0 else ""]
    dist.broadcast_object_list(ts_list, src=0)
    timestamp = ts_list[0]

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds = BigEarthNetDataset(
        cfg["data"]["root"], cfg["data"]["metadata"], split="train",
        transform=get_transforms("train"),
    )
    val_ds = BigEarthNetDataset(
        cfg["data"]["root"], cfg["data"]["metadata"], split="val",
        transform=get_transforms("val"),
    )

    if rank == 0:
        print(f"Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")
        print(f"Dispositivo : {device}  |  rank {rank}/{world_size}")
        print(f"Batch/GPU   : {cfg['training']['batch_size']}  → global batch: "
              f"{cfg['training']['batch_size'] * world_size}")

    # ── DistributedSampler ───────────────────────────────────────────────────
    # drop_last=True on train keeps equal-sized batches across ranks.
    # drop_last=False on val preserves all validation samples for unbiased metrics.
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=train_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=val_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # ── Build trainer stack ──────────────────────────────────────────────────
    # Non-zero ranks get --trace off so only rank 0 writes logs and plots.
    metrics = args.metrics if args.metrics is not None else []
    layers = args.layers or []
    fn = args.fn or []
    trace = args.trace if rank == 0 else "off"
    layers_r0 = layers if rank == 0 else []

    builder = (
        TrainingSessionBuilder(cfg, device, timestamp, rank=rank, world_size=world_size, distributed=True)
        .with_trace(trace)
        .with_layers(*layers_r0)
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
        print(f"Modelo      : {model_name}")
        print(f"Traza       : {trace}")
        print(f"Capas       : {layers or 'ninguna'}")
        print(f"Decoradores@: {fn or 'ninguno'}")

    # ── Entrenamiento ────────────────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    trainer.fit(train_loader, val_loader, epochs=epochs)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
