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

--metrics [loss] [f1] [accuracy] [precision_recall]
    Metric reporters individuales (solo para --trace off/simple):
      loss             — LossReporter: train_loss / val_loss
      f1               — F1Reporter: train_f1 / val_f1
      accuracy         — AccuracyReporter: train_acc / val_acc
      precision_recall — PrecisionRecallReporter: val_precision / val_recall
    Sin args (--metrics sin valores) desactiva todos los reporters.
    Con --trace deep este flag se ignora (DeepTracingDecorator los gestiona).
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
    LossReporter,
    F1Reporter,
    AccuracyReporter,
    PrecisionRecallReporter,
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
    parser.add_argument(
        "--metrics", nargs="*",
        choices=["loss", "f1", "accuracy", "precision_recall"],
        default=["loss", "f1", "accuracy", "precision_recall"],
        help=(
            "Metric reporters individuales (solo para --trace off/simple). "
            "Sin args (--metrics) desactiva todos. "
            "Choices: loss f1 accuracy precision_recall"
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

    print(f"Dispositivo : {device}")
    print(f"Traza       : {args.trace}")
    print(f"Capas       : {args.layers or 'ninguna'}")
    print(f"Decoradores@: {args.fn or 'ninguno'}")
    print(f"Métricas    : {metrics or 'ninguna (--trace deep las gestiona internamente)'}")

    if args.trace == "deep":
        if "hooks" in (args.layers or []):
            print("  [aviso] --layers hooks ignorado con --trace deep "
                  "(DeepTracingDecorator ya registra sus propios hooks)")
        if args.fn:
            print("  [aviso] --fn aplicado solo a eval_epoch con --trace deep "
                  "(train_epoch es gestionado directamente por DeepTracingDecorator)")

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
    #     └── metric reporter N   ← --metrics   (solo para --trace off/simple)
    #           └── metric reporter 1
    #                 └── aspecto N   ← --layers
    #                       └── aspecto 1
    #                             └── Trainer  ← métodos con @ si --fn activo
    #
    # Orden de construcción (de dentro hacia afuera):
    #   1. Decoradores @  sobre métodos del Trainer base
    #   2. Decoradores de aspecto OOP (hooks, plot)
    #   3. Metric reporters OOP (loss, f1, accuracy, precision_recall)
    #   4. Controlador OOP (el más externo)

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

    # 2. Decoradores de aspecto OOP
    inner = base
    layers = args.layers or []
    if "hooks" in layers:
        inner = LayerHooksDecorator(inner)
    if "plot" in layers:
        inner = PlottingDecorator(inner, output_path=f"plots/training_{timestamp}.png")

    # 3. Logger (necesario antes de crear reporters y controlador)
    logger = None
    if args.trace in ("simple", "deep"):
        log_file = (
            f"logs/train_deep_{timestamp}.log"
            if args.trace == "deep"
            else f"logs/train_{timestamp}.log"
        )
        logger = setup_logger("trainer", log_file=log_file)

    # 4. Metric reporters (solo para --trace off/simple; deep los gestiona internamente)
    #    Se apilan de dentro hacia afuera: el primero en imprimir es el más interno.
    #    Orden de salida: loss → f1 → accuracy → precision_recall
    if args.trace != "deep":
        if "loss" in metrics:
            inner = LossReporter(inner, logger=logger)
        if "f1" in metrics:
            inner = F1Reporter(inner, logger=logger)
        if "accuracy" in metrics:
            inner = AccuracyReporter(inner, logger=logger)
        if "precision_recall" in metrics:
            inner = PrecisionRecallReporter(inner, logger=logger)

    # 5. Controlador OOP (siempre el más externo)
    if args.trace == "off":
        trainer = TracingDecorator(inner)
    elif args.trace == "simple":
        trainer = TracingDecorator(inner, logger=logger)
    else:  # deep
        trainer = DeepTracingDecorator(
            inner, logger=logger,
            log_every=cfg["training"].get("log_batch_every", 100),
        )

    # ── Entrenamiento ────────────────────────────────────────────────────────

    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])


if __name__ == "__main__":
    main()
