"""TrainingSessionBuilder — fluent builder for the full trainer + decorator stack.

Encapsulates all the wiring logic that previously lived in train_single_gpu.py:
model construction, optimizer selection (LLRD vs AdamW), scheduler (warmup + cosine),
logger setup, and the ordered stacking of OOP decorators and @ function decorators.

Usage:
    trainer = (
        TrainingSessionBuilder(cfg, device, timestamp)
        .with_model("vit_tiny_patch16_224")   # optional override
        .with_trace("simple")
        .with_layers("plot", "hooks", "confusion", "batch-monitor")
        .with_fn("energy")
        .with_metrics("loss", "f1", "accuracy", "precision_recall")
        .with_inspect("model-summary", "grad-monitor", "anomalies")
        .build()
    )
    trainer.fit(train_loader, val_loader, epochs=epochs)
"""

import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from src.training.base_trainer import BaseTrainer
from src.training.trainer import Trainer
from src.training.logger_setup import setup_logger
from src.training.fn_decorators import timed, measure_energy
from src.training.decorators import (
    TracingDecorator,
    DeepTracingDecorator,
    ALL_INSPECT_FEATURES,
    PlottingDecorator,
    LayerHooksDecorator,
    ConfusionMatrixDecorator,
    BatchMonitorDecorator,
    LossReporter,
    F1Reporter,
    AccuracyReporter,
    PrecisionRecallReporter,
)
from src.models.vit import build_model, build_llrd_optimizer


class TrainingSessionBuilder:
    """Fluent builder for the Trainer + decorator stack.

    Call .with_*() methods to configure the session, then .build() to get
    a fully wired trainer ready for .fit().

    The decorator stack is always assembled in this fixed order (inner → outer):
      Trainer
        ← @ function decorators (timed, measure_energy)
        ← aspect decorators    (hooks, plot, confusion, batch-monitor)
        ← metric reporters     (loss, f1, accuracy, precision_recall)
        ← controller           (TracingDecorator or DeepTracingDecorator)
    """

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        timestamp: str | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        self._cfg = cfg
        self._device = device
        self._timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._rank = rank
        self._world_size = world_size

        # Defaults — mirrors the CLI defaults in train_single_gpu.py
        self._model_name: str | None = None          # None → use cfg["model"]["name"]
        self._trace: str = "simple"
        self._layers: list[str] = []
        self._fn: list[str] = []
        self._metrics: list[str] = ["loss", "f1", "accuracy", "precision_recall"]
        self._inspect: set[str] | None = None        # None → not active

    # ── Fluent configuration API ─────────────────────────────────────────────

    def with_model(self, model_name: str) -> "TrainingSessionBuilder":
        """Override the model name from the config."""
        self._model_name = model_name
        return self

    def with_trace(self, mode: str) -> "TrainingSessionBuilder":
        """Set the logging controller: 'off', 'simple', or 'deep'."""
        if mode not in ("off", "simple", "deep"):
            raise ValueError(f"trace must be 'off', 'simple' or 'deep', got '{mode}'")
        self._trace = mode
        return self

    def with_layers(self, *layers: str) -> "TrainingSessionBuilder":
        """Add aspect decorators: 'plot', 'hooks', 'confusion', 'batch-monitor'."""
        self._layers = list(layers)
        return self

    def with_fn(self, *fns: str) -> "TrainingSessionBuilder":
        """Add @ function decorators: 'timing', 'energy'."""
        self._fn = list(fns)
        return self

    def with_metrics(self, *metrics: str) -> "TrainingSessionBuilder":
        """Select metric reporters (only active for --trace off/simple):
        'loss', 'f1', 'accuracy', 'precision_recall'. Pass nothing to disable all."""
        self._metrics = list(metrics)
        return self

    def with_inspect(self, *features: str) -> "TrainingSessionBuilder":
        """Enable modular inspection features (activates DeepTracingDecorator):
        'model-summary', 'grad-monitor', 'batch-table', 'anomalies'."""
        self._inspect = set(features)
        return self

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self) -> BaseTrainer:
        """Assemble and return the fully configured trainer stack."""
        cfg = self._cfg

        # ── Model ────────────────────────────────────────────────────────────
        model_name = self._model_name or cfg["model"]["name"]
        model = build_model(
            model_name=model_name,
            num_classes=cfg["model"].get("num_classes", 19),
            pretrained=cfg["model"].get("pretrained", True),
        )

        # ── Optimizer ────────────────────────────────────────────────────────
        lr_base = cfg["training"]["lr"]
        weight_decay = cfg["training"].get("weight_decay", 0.05)
        llrd_decay = cfg["training"].get("llrd_decay", 0.0)

        if llrd_decay > 0.0 and model.is_vit:
            optimizer = build_llrd_optimizer(model, lr_base, weight_decay, llrd_decay)
        else:
            if llrd_decay > 0.0 and not model.is_vit:
                print(f"  [aviso] LLRD no disponible para '{model_name}' (CNN). Usando AdamW estándar.")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr_base, weight_decay=weight_decay
            )

        # ── Scheduler ────────────────────────────────────────────────────────
        epochs = cfg["training"]["epochs"]
        lr_min = cfg["training"].get("lr_min", 1e-6)
        warmup_epochs = cfg["training"].get("warmup_epochs", 0)

        if warmup_epochs > 0:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
            )
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=lr_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr_min,
            )

        # ── Base Trainer ──────────────────────────────────────────────────────
        grad_clip = cfg["training"].get("grad_clip", None)
        if self._world_size > 1:
            from src.training.ddp_trainer import DDPTrainer
            base = DDPTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self._device,
                checkpoint_dir=cfg["checkpoint"]["dir"],
                grad_clip=grad_clip,
                rank=self._rank,
                world_size=self._world_size,
            )
        else:
            base = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self._device,
                checkpoint_dir=cfg["checkpoint"]["dir"],
                grad_clip=grad_clip,
            )

        # ── 1. @ function decorators on Trainer methods ───────────────────────
        if "energy" in self._fn:
            base.train_epoch = measure_energy(base.train_epoch)
            base.eval_epoch = measure_energy(base.eval_epoch)
        if "timing" in self._fn:
            base.train_epoch = timed(base.train_epoch)
            base.eval_epoch = timed(base.eval_epoch)

        # ── 2. Aspect decorators (inner → outer) ──────────────────────────────
        inner: BaseTrainer = base
        if "hooks" in self._layers:
            inner = LayerHooksDecorator(inner)
        if "batch-monitor" in self._layers:
            inner = BatchMonitorDecorator(
                inner,
                log_every=cfg["training"].get("log_batch_every", 50),
                output_dir="logs",
                timestamp=self._timestamp,
            )
        if "plot" in self._layers:
            inner = PlottingDecorator(
                inner, output_path=f"plots/training_{self._timestamp}.png"
            )
        if "confusion" in self._layers:
            inner = ConfusionMatrixDecorator(
                inner, output_dir="plots", timestamp=self._timestamp
            )

        # ── 3. Logger ─────────────────────────────────────────────────────────
        logger: logging.Logger | None = None
        use_deep = self._trace == "deep" or self._inspect is not None

        if self._trace in ("simple", "deep") or use_deep:
            prefix = "train_deep" if use_deep else "train"
            log_file = f"logs/{prefix}_{self._timestamp}.log"
            logger = setup_logger("trainer", log_file=log_file)

        # ── 4. Metric reporters (only when not using DeepTracingDecorator) ────
        if not use_deep:
            if "loss" in self._metrics:
                inner = LossReporter(inner, logger=logger)
            if "f1" in self._metrics:
                inner = F1Reporter(inner, logger=logger)
            if "accuracy" in self._metrics:
                inner = AccuracyReporter(inner, logger=logger)
            if "precision_recall" in self._metrics:
                inner = PrecisionRecallReporter(inner, logger=logger)

        # ── 5. Controller (outermost) ─────────────────────────────────────────
        patience = cfg["training"].get("early_stopping_patience", None)

        if use_deep:
            features = (
                self._inspect
                if self._inspect is not None
                else set(ALL_INSPECT_FEATURES)  # --trace deep → all features
            )
            trainer = DeepTracingDecorator(
                inner,
                logger=logger,
                log_every=cfg["training"].get("log_batch_every", 100),
                patience=patience,
                features=features,
            )
        elif self._trace == "off":
            trainer = TracingDecorator(inner, patience=patience)
        else:  # simple
            trainer = TracingDecorator(inner, logger=logger, patience=patience)

        return trainer

    # ── Informational summary ─────────────────────────────────────────────────

    def print_config(self, model_params: int | None = None):
        """Print a summary of the configured session to stdout."""
        model_name = self._model_name or self._cfg["model"]["name"]
        use_deep = self._trace == "deep" or self._inspect is not None
        features_str = (
            ", ".join(sorted(self._inspect)) if self._inspect is not None
            else ("todas" if self._trace == "deep" else "ninguna")
        )
        print(f"Modelo      : {model_name}" + (f" | Parámetros: {model_params:,}" if model_params else ""))
        print(f"Traza       : {self._trace}" + (" (DeepTracingDecorator activo)" if use_deep else ""))
        print(f"Inspect     : {features_str}")
        print(f"Capas       : {self._layers or 'ninguna'}")
        print(f"Decoradores@: {self._fn or 'ninguno'}")
        print(f"Métricas    : {self._metrics or 'ninguna'}")
