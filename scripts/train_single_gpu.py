"""Single-GPU training script for BigEarthNet-S2.

Flags
-----
--trace off|simple|deep
    Logging controller:
      off    — TracingDecorator, console only
      simple — TracingDecorator, log to logs/train_FECHA.log
      deep   — DeepTracingDecorator with all inspection features active
               (equivalent to --inspect model-summary grad-monitor batch-table anomalies)

--inspect [model-summary] [grad-monitor] [batch-table] [anomalies]
    Activates DeepTracingDecorator with only the selected features:
      model-summary  — torchinfo architecture summary at fit start
      grad-monitor   — forward/backward/param hooks on every layer
      batch-table    — per-layer stats table every log_batch_every batches
      anomalies      — dead-neuron / vanishing / exploding gradient alerts
    If --inspect is used, --trace deep is implied (DeepTracingDecorator becomes controller).

--layers [plot] [hooks] [confusion] [batch-monitor]
    Stackable aspect decorators:
      plot         — PlottingDecorator: saves loss+F1 PNG to plots/ after each epoch
      hooks        — LayerHooksDecorator: forward hooks on Linear layers every 5 epochs
      confusion    — ConfusionMatrixDecorator: per-class F1/precision/recall bar charts
      batch-monitor — BatchMonitorDecorator: batch-level loss CSV in logs/

--fn [timing] [energy]
    Python @ decorators applied to train_epoch and eval_epoch:
      timing — print execution time
      energy — measure GPU energy consumption (requires nvidia-ml-py)

--metrics [loss] [f1] [accuracy] [precision_recall]
    Metric reporters (active only when --inspect is NOT used):
      loss             — LossReporter: train_loss / val_loss
      f1               — F1Reporter: train_f1 / val_f1
      accuracy         — AccuracyReporter: train_acc / val_acc
      precision_recall — PrecisionRecallReporter: val_precision / val_recall
    Pass --metrics with no args to disable all reporters.

--model <name>
    Override model name from config (any timm model ID):
      vit_tiny_patch16_224    — ~5.7M params, fast for local testing
      vit_small_patch16_224   — ~22M params
      vit_base_patch16_224    — ~85M params (default)
      resnet50                — ~25M params, CNN (no LLRD)
      efficientnet_b0         — ~5.3M params, CNN (no LLRD)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BigEarthNetDataset, get_transforms
from src.training.builder import TrainingSessionBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Train on BigEarthNet-S2 (single GPU)")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data-root", type=str, help="Override data.root")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size")
    parser.add_argument("--model", type=str, help="Override model name (any timm ID)")
    parser.add_argument(
        "--trace", choices=["off", "simple", "deep"], default="simple",
        help="Logging controller mode",
    )
    parser.add_argument(
        "--inspect", nargs="*",
        choices=["model-summary", "grad-monitor", "batch-table", "anomalies"],
        default=None,
        help="Modular inspection features (activates DeepTracingDecorator)",
    )
    parser.add_argument(
        "--layers", nargs="*",
        choices=["plot", "hooks", "confusion", "batch-monitor"],
        default=[],
        help="Stackable aspect decorators",
    )
    parser.add_argument(
        "--fn", nargs="*", choices=["timing", "energy"], default=[],
        help="Python @ decorators to apply to train_epoch / eval_epoch",
    )
    parser.add_argument(
        "--metrics", nargs="*",
        choices=["loss", "f1", "accuracy", "precision_recall"],
        default=["loss", "f1", "accuracy", "precision_recall"],
        help=(
            "Metric reporters (only active without --inspect). "
            "Pass --metrics with no args to disable all."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics = args.metrics if args.metrics is not None else []
    layers = args.layers or []
    fn = args.fn or []

    # ── Datos ────────────────────────────────────────────────────────────────

    train_ds = BigEarthNetDataset(
        cfg["data"]["root"], cfg["data"]["metadata"], split="train",
        transform=get_transforms("train"),
    )
    val_ds = BigEarthNetDataset(
        cfg["data"]["root"], cfg["data"]["metadata"], split="val",
        transform=get_transforms("val"),
    )
    print(f"Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    # ── Construcción del stack via Builder ───────────────────────────────────

    builder = (
        TrainingSessionBuilder(cfg, device, timestamp)
        .with_trace(args.trace)
        .with_layers(*layers)
        .with_fn(*fn)
        .with_metrics(*metrics)
    )
    if args.model:
        builder = builder.with_model(args.model)
    if args.inspect is not None:
        builder = builder.with_inspect(*args.inspect)

    trainer = builder.build()

    # Print summary after build (model already instantiated inside builder)
    model_name = args.model or cfg["model"]["name"]
    use_deep = args.trace == "deep" or args.inspect is not None
    print(f"Dispositivo : {device}")
    print(f"Modelo      : {model_name}")
    print(f"Traza       : {args.trace}" + (" [DeepTracingDecorator]" if use_deep else ""))
    print(f"Inspect     : {sorted(args.inspect) if args.inspect is not None else ('todas' if args.trace == 'deep' else 'ninguna')}")
    print(f"Capas       : {layers or 'ninguna'}")
    print(f"Decoradores@: {fn or 'ninguno'}")
    print(f"Métricas    : {metrics or 'ninguna (DeepTracingDecorator las gestiona)'}")

    # ── Entrenamiento ────────────────────────────────────────────────────────

    epochs = cfg["training"]["epochs"]
    trainer.fit(train_loader, val_loader, epochs=epochs)


if __name__ == "__main__":
    main()
