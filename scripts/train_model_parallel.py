"""Model-parallel (pipeline) training for BigEarthNet-S2.

Splits a ViT across two devices (see ``src/models/model_parallel.py``) and trains
it through the SAME stack as every other strategy: the Template-Method loop in
``EpochController`` and the full decorator stack, assembled by
``TrainingSessionBuilder.with_model_parallel(...)``. So a model-parallel run gets
the same flags and the same artifacts (curves, per-class, confusion, batch metrics,
energy) as single/DDP — it is a first-class strategy, not a bespoke loop.

Flags (same meaning as train_single_gpu.py):
  --trace off|simple                 logging controller (deep needs a single-device model)
  --layers plot hooks confusion batch-monitor   stackable aspect decorators
  --fn timing energy                 @ decorators (energy samples BOTH split GPUs)
  --metrics loss f1 accuracy precision_recall   metric reporters
Model-parallel specific:
  --devices cuda:0,cuda:1            where to place the two stages (auto-detected if omitted)
  --split-block N                    block index where stage 0 ends (default: half)
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BigEarthNetDataset, get_transforms
from src.training.builder import TrainingSessionBuilder
from src.training.reproducibility import set_seed, make_generator, seed_worker


def parse_args():
    p = argparse.ArgumentParser(description="Model-parallel training for BigEarthNet-S2")
    p.add_argument("--config", type=str, default="configs/train_model_parallel_kaggle.yaml")
    p.add_argument("--model", type=str, default=None, help="Override model name (timm ViT)")
    p.add_argument("--precision", choices=["fp32", "tf32", "amp", "bf16"], default=None,
                   help="Override training.precision (Tensor cores)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--split-block", type=int, default=None,
                   help="Block index where stage 0 ends (default: half the blocks)")
    p.add_argument("--devices", type=str, default=None,
                   help="Comma-separated device list, e.g. 'cuda:0,cuda:1'. Auto-detected if omitted.")
    p.add_argument("--trace", choices=["off", "simple"], default="simple",
                   help="Logging controller (deep tracing needs a single-device model)")
    p.add_argument("--layers", nargs="*", choices=["plot", "hooks", "confusion", "batch-monitor"],
                   default=[], help="Stackable aspect decorators")
    p.add_argument("--fn", nargs="*", choices=["timing", "energy"], default=[],
                   help="@ decorators; 'energy' samples the power of BOTH split GPUs")
    p.add_argument("--metrics", nargs="*",
                   choices=["loss", "f1", "accuracy", "precision_recall"],
                   default=["loss", "f1", "accuracy", "precision_recall"],
                   help="Metric reporters (pass --metrics with no args to disable all)")
    p.add_argument("--batch-log-every", type=int, default=None, metavar="N",
                   help="Log batch metrics every N batches (with --layers batch-monitor)")
    return p.parse_args()


def _pick_devices(arg: str | None) -> list[str]:
    if arg:
        return [d.strip() for d in arg.split(",")]
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n >= 2:
        return ["cuda:0", "cuda:1"]
    if n == 1:
        return ["cuda:0", "cuda:0"]
    return ["cpu", "cpu"]


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.precision:
        cfg["training"]["precision"] = args.precision
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    model_name = cfg["model"]["name"]
    devices = _pick_devices(args.devices)
    distinct = len(set(devices)) > 1
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    # ── Reproducibility (mirror train_single_gpu.py) ──────────────────────────────
    seed = cfg["training"].get("seed")
    loader_generator, worker_init = None, None
    if seed is not None:
        set_seed(int(seed))
        loader_generator = make_generator(int(seed))
        worker_init = seed_worker
        print(f"Semilla     : {seed} (run determinista)")

    metrics = args.metrics if args.metrics is not None else []
    layers = args.layers or []
    fn = args.fn or []

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = BigEarthNetDataset(cfg["data"]["root"], cfg["data"]["metadata"],
                                  split="train", transform=get_transforms("train"))
    val_ds = BigEarthNetDataset(cfg["data"]["root"], cfg["data"]["metadata"],
                                split="val", transform=get_transforms("val"))
    bs = cfg["training"]["batch_size"]
    nw = cfg["data"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              pin_memory=True, generator=loader_generator, worker_init_fn=worker_init)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=True, worker_init_fn=worker_init)
    print(f"Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")

    # ── Build the stack via the Builder (model split + full decorator stack) ────
    builder = (
        TrainingSessionBuilder(cfg, torch.device(devices[-1]), timestamp)
        .with_model_parallel(devices, args.split_block)
        .with_trace(args.trace)
        .with_layers(*layers)
        .with_fn(*fn)
        .with_metrics(*metrics)
    )
    if args.model:
        builder = builder.with_model(args.model)
    if args.batch_log_every is not None:
        builder = builder.with_batch_log_every(args.batch_log_every)
    trainer = builder.build()

    print(f"Modelo      : {model_name} (paralelismo de modelo)")
    print(f"Devices     : {devices} "
          f"({'reparto real en 2 dispositivos' if distinct else 'un solo dispositivo — demo'})")
    print(f"Traza       : {args.trace} | Capas: {layers or 'ninguna'} | "
          f"Decoradores@: {fn or 'ninguno'} | Métricas: {metrics or 'ninguna'}")

    # Config line for the dashboard's Run -> Info panel.
    _prec = cfg["training"].get("precision", "fp32")
    _loss = str(cfg["training"].get("loss", "bce")).lower()
    logging.getLogger("trainer").info(
        f"Configuración: modelo={model_name} | paralelismo=modelo(pipeline) | "
        f"devices={'+'.join(devices)} | split_block={args.split_block or 'auto'} | "
        f"batch={bs} (global) | epochs={cfg['training']['epochs']} | "
        f"lr={cfg['training']['lr']} | precision={_prec} | loss={_loss} | "
        f"train={len(train_ds)} | val={len(val_ds)}"
    )

    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])


if __name__ == "__main__":
    main()
