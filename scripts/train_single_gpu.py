"""Single-GPU training script for BigEarthNet-S2.

Modos de traza (--trace):
  off    — solo métricas por epoch, sin overhead (entrenamiento real)
  simple — igual + timestamps + log a fichero
  deep   — trazado completo por capa y parámetro + log a fichero
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
from src.models.vit import build_model
from src.training.trainer import Trainer
from src.training.logger_setup import setup_logger
from src.training.trainer_decorators import MetricsLoggerDecorator, TracingDecorator
from src.training.deep_tracing import DeepTracingDecorator


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT on BigEarthNet-S2 (single GPU)")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data-root", type=str, help="Override data.root")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size")
    parser.add_argument(
        "--trace",
        choices=["off", "simple", "deep"],
        default="simple",
        help="Nivel de traza: off=solo métricas, simple=timestamps+fichero, deep=trazado por capa",
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
    print(f"Dispositivo: {device}  |  Modo traza: {args.trace}")

    # Datasets
    train_dataset = BigEarthNetDataset(
        cfg["data"]["root"],
        metadata_path=cfg["data"]["metadata"],
        split="train",
        transform=get_transforms("train"),
    )
    val_dataset = BigEarthNetDataset(
        cfg["data"]["root"],
        metadata_path=cfg["data"]["metadata"],
        split="val",
        transform=get_transforms("val"),
    )
    print(f"Train: {len(train_dataset)} patches | Val: {len(val_dataset)} patches")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # Modelo
    model = build_model(
        model_name=cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
    )
    print(f"Modelo: {cfg['model']['name']} | Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer y scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    # Trainer base
    base = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=cfg["checkpoint"]["dir"],
    )

    # Stack de decoradores según --trace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.trace == "off":
        trainer = MetricsLoggerDecorator(base)

    elif args.trace == "simple":
        logger = setup_logger(
            name="trainer",
            log_file=f"logs/train_{timestamp}.log",
        )
        trainer = TracingDecorator(base, logger=logger)

    else:  # deep
        logger = setup_logger(
            name="trainer",
            log_file=f"logs/train_deep_{timestamp}.log",
        )
        trainer = DeepTracingDecorator(
            base,
            logger=logger,
            log_every=cfg["training"].get("log_batch_every", 100),
        )

    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])


if __name__ == "__main__":
    main()
