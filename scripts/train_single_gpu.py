"""Single-GPU training script for BigEarthNet-S2.

Flags
-----
--trace off|simple|deep
    Elige el decorador controlador del bucle de training:
      off    — TracingDecorator sin fichero (solo consola)
      simple — TracingDecorator con log a fichero logs/train_FECHA.log
      deep   — DeepTracingDecorator: trazado completo por capa y parámetro

--layers [plot] [hooks]
    Decoradores de aspecto apilables entre el Trainer y el controlador:
      plot  — PlottingDecorator: guarda curvas PNG en plots/ tras cada epoch
      hooks — LayerHooksDecorator: activa forward hooks cada 5 epochs

--fn [timing] [energy]
    Decoradores @ de Python aplicados a train_epoch y eval_epoch:
      timing — imprime tiempo de ejecución de cada método
      energy — mide consumo energético GPU por epoch (requiere nvidia-ml-py)
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
from src.training.decorators import (
    TracingDecorator,
    DeepTracingDecorator,
    PlottingDecorator,
    LayerHooksDecorator,
)
from src.training.fn_decorators import timed, measure_energy


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT on BigEarthNet-S2 (single GPU)")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data-root", type=str, help="Override data.root")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size")
    parser.add_argument(
        "--trace", choices=["off", "simple", "deep"], default="simple",
        help="Controlador de logging: off=consola, simple=fichero, deep=trazado por capa",
    )
    parser.add_argument(
        "--layers", nargs="*", choices=["plot", "hooks"], default=[],
        help="Decoradores de aspecto a apilar (combinables): plot hooks",
    )
    parser.add_argument(
        "--fn", nargs="*", choices=["timing", "energy"], default=[],
        help="Decoradores @ de Python a aplicar: timing energy",
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

    print(f"Dispositivo : {device}")
    print(f"Traza       : {args.trace}")
    print(f"Capas       : {args.layers or 'ninguna'}")
    print(f"Decoradores@: {args.fn or 'ninguno'}")

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

    # ── Modelo ───────────────────────────────────────────────────────────────

    model = build_model(cfg["model"]["name"], pretrained=cfg["model"]["pretrained"])
    print(f"Modelo: {cfg['model']['name']} | Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"],
    )

    # ── Construcción del stack de decoradores ────────────────────────────────
    #
    #   OUTER (controlador)
    #     └── aspecto N   ← --layers
    #           └── aspecto 1
    #                 └── Trainer  ← métodos decorados con @ si --fn activo
    #
    # Los decoradores @ se aplican primero (sobre los métodos del Trainer base).
    # Los decoradores de aspecto (OOP) se apilan encima.
    # El controlador (OOP) va siempre en el exterior.

    base = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        device=device, checkpoint_dir=cfg["checkpoint"]["dir"],
    )

    # 1. Decoradores @ de Python (sobre métodos concretos del Trainer)
    fn = args.fn or []
    if "energy" in fn:
        base.train_epoch = measure_energy(base.train_epoch)
        base.eval_epoch = measure_energy(base.eval_epoch)
    if "timing" in fn:
        base.train_epoch = timed(base.train_epoch)
        base.eval_epoch = timed(base.eval_epoch)

    # 2. Decoradores de aspecto OOP (se apilan sobre el Trainer)
    inner = base
    layers = args.layers or []
    if "hooks" in layers:
        inner = LayerHooksDecorator(inner)
    if "plot" in layers:
        inner = PlottingDecorator(inner, output_path=f"plots/training_{timestamp}.png")

    # 3. Controlador OOP (siempre el más externo)
    if args.trace == "off":
        trainer = TracingDecorator(inner)

    elif args.trace == "simple":
        logger = setup_logger("trainer", log_file=f"logs/train_{timestamp}.log")
        trainer = TracingDecorator(inner, logger=logger)

    else:  # deep
        logger = setup_logger("trainer", log_file=f"logs/train_deep_{timestamp}.log")
        trainer = DeepTracingDecorator(
            inner, logger=logger,
            log_every=cfg["training"].get("log_batch_every", 100),
        )

    # ── Entrenamiento ────────────────────────────────────────────────────────

    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])


if __name__ == "__main__":
    main()
