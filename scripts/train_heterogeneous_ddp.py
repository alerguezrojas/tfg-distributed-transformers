"""Heterogeneous DDP training — mixed GPU + CPU across nodes.

Allows training on heterogeneous hardware (e.g. V100 GPU + CPU nodes) by:
  1. Assigning proportional data fractions per rank via HeterogeneousDistributedSampler
  2. Normalizing gradients by global batch size via HeterogeneousDDPTrainer

This solves the problem that verode16/18 GPUs are incompatible with PyTorch 2.x
while their CPUs are still usable as auxiliary workers.

Launch (run both commands at the same time in separate tmux windows):

    # verode21 (GPU, rank 0):
    ssh verode21
    cd ~/tfg-distributed-transformers
    .venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \\
      --master_addr=verode21 --master_port=29501 \\
      scripts/train_heterogeneous_ddp.py \\
      --config configs/train_heterogeneous_ddp.yaml --trace simple

    # verode16 or verode18 (CPU, rank 1):
    ssh verode16
    cd ~/tfg-distributed-transformers
    .venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \\
      --master_addr=verode21 --master_port=29501 \\
      scripts/train_heterogeneous_ddp.py \\
      --config configs/train_heterogeneous_ddp.yaml --trace simple

The gloo backend is used for all ranks (supports both CPU and GPU).
Only rank 0 writes logs and checkpoints.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BigEarthNetDataset, get_transforms
from src.training.builder import TrainingSessionBuilder
from src.training.heterogeneous_sampler import HeterogeneousDistributedSampler
from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Heterogeneous DDP training (GPU + CPU nodes)"
    )
    parser.add_argument("--config", type=str,
                        default="configs/train_heterogeneous_ddp.yaml")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--model", type=str, help="Override model name")
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
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dist_cfg = cfg.get("distributed", {})
    backend = dist_cfg.get("backend", "gloo")
    ranks_cfg = dist_cfg.get("ranks", [])

    # Determine this rank's device from config
    local_rank = int(os.environ["LOCAL_RANK"])
    rank_info = ranks_cfg[int(os.environ.get("RANK", local_rank))] if ranks_cfg else {}
    rank_device_type = rank_info.get("device", "cpu")

    if rank_device_type == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    # Broadcast timestamp from rank 0
    ts_list = [datetime.now().strftime("%d%m%Y_%H%M%S") if rank == 0 else ""]
    dist.broadcast_object_list(ts_list, src=0)
    timestamp = ts_list[0]

    # Per-rank config
    if ranks_cfg and rank < len(ranks_cfg):
        rank_batch_size = ranks_cfg[rank].get("batch_size", cfg["training"]["batch_size"])
        compute_weights = [r.get("compute_weight", 1) for r in ranks_cfg]
    else:
        rank_batch_size = cfg["training"]["batch_size"]
        compute_weights = [1] * world_size

    global_batch_size = sum(
        (ranks_cfg[r].get("batch_size", cfg["training"]["batch_size"])
         if ranks_cfg and r < len(ranks_cfg) else cfg["training"]["batch_size"])
        for r in range(world_size)
    )

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
        print(f"Device rank {rank}: {device}")
        print(f"Global batch size: {global_batch_size} "
              f"(rank batch sizes: {[r.get('batch_size', cfg['training']['batch_size']) for r in ranks_cfg]})")
        print(f"Compute weights: {compute_weights}")

    # ── Heterogeneous samplers ────────────────────────────────────────────────
    train_sampler = HeterogeneousDistributedSampler(
        train_ds, weights=compute_weights, rank=rank, world_size=world_size,
        shuffle=True, drop_last=True,
    )
    # Validation: use standard equal split (metrics gathered from all ranks)
    from torch.utils.data import DistributedSampler
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

    # ── Build trainer stack ───────────────────────────────────────────────────
    # Override batch_size in cfg so builder knows the per-rank size
    cfg["training"]["batch_size"] = rank_batch_size

    trace = args.trace if rank == 0 else "off"
    layers = (args.layers or []) if rank == 0 else []
    fn = args.fn or []

    builder = (
        TrainingSessionBuilder(
            cfg, device, timestamp, rank=rank, world_size=world_size,
            distributed=True,
        )
        .with_trace(trace)
        .with_layers(*layers)
        .with_fn(*fn)
        .with_metrics(*(args.metrics or []))
    )
    if args.model:
        builder = builder.with_model(args.model)

    # Build base trainer then replace the inner trainer with HeterogeneousDDPTrainer
    trainer = builder.build()

    # Unwrap to find the innermost Trainer and replace it
    # The builder already created a DDPTrainer — we patch local_batch_size
    def _find_ddp(t):
        if isinstance(t, HeterogeneousDDPTrainer):
            return t
        if hasattr(t, "_trainer"):
            return _find_ddp(t._trainer)
        return None

    # Re-build with HeterogeneousDDPTrainer by instantiating it directly
    inner = _find_core(trainer)
    hetero_trainer = HeterogeneousDDPTrainer(
        model=inner.model.module if hasattr(inner.model, "module") else inner.model,
        optimizer=inner.optimizer,
        scheduler=inner.scheduler,
        device=device,
        checkpoint_dir=inner.checkpoint_dir,
        criterion=inner.criterion,
        grad_clip=inner.grad_clip,
        label_smoothing=inner.label_smoothing,
        mixup_alpha=inner.mixup_alpha,
        rank=rank,
        world_size=world_size,
        local_batch_size=rank_batch_size,
    )

    # Re-wrap with the decorator stack from the builder
    trainer = _rewrap(trainer, inner, hetero_trainer)

    if rank == 0:
        model_name = args.model or cfg["model"]["name"]
        print(f"Model: {model_name} | Trace: {trace} | Layers: {layers or 'none'}")
        print(f"Heterogeneous DDP active — rank {rank} batch_size={rank_batch_size}")

    # ── Train ─────────────────────────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    trainer.fit(train_loader, val_loader, epochs=epochs)

    dist.destroy_process_group()


def _find_core(trainer):
    """Walk the decorator stack to find the innermost DDPTrainer/Trainer."""
    from src.training.ddp_trainer import DDPTrainer
    t = trainer
    while hasattr(t, "_trainer"):
        t = t._trainer
    return t


def _rewrap(outer, old_inner, new_inner):
    """Replace old_inner with new_inner in the decorator stack."""
    from src.training.ddp_trainer import DDPTrainer

    def _replace(t):
        if t is old_inner:
            return new_inner
        if hasattr(t, "_trainer"):
            t._trainer = _replace(t._trainer)
        return t

    return _replace(outer)


if __name__ == "__main__":
    main()
