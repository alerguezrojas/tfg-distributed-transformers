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
        distributed: bool = False,
    ):
        self._cfg = cfg
        self._device = device
        self._timestamp = timestamp or datetime.now().strftime("%d%m%Y_%H%M%S")
        self._rank = rank
        self._world_size = world_size
        self._distributed = distributed

        # Defaults — mirrors the CLI defaults in train_single_gpu.py
        self._model_name: str | None = None          # None → use cfg["model"]["name"]
        self._trace: str = "simple"
        self._layers: list[str] = []
        self._fn: list[str] = []
        self._metrics: list[str] = ["loss", "f1", "accuracy", "precision_recall"]
        self._inspect: set[str] | None = None        # None → not active
        self._batch_log_every: int | None = None     # None → use cfg or default
        self._hetero_local_batch_size: int | None = None  # None → standard DDPTrainer
        self._model_parallel: tuple | None = None     # None → not model-parallel; else (devices, split_block)
        self._output_mode: str | None = None         # None → "ddp"/"single" por defecto

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

    def with_batch_log_every(self, n: int) -> "TrainingSessionBuilder":
        """Log batch metrics every N batches (default: 1 = every batch)."""
        self._batch_log_every = max(1, n)
        return self

    def with_heterogeneous_ddp(self, local_batch_size: int) -> "TrainingSessionBuilder":
        """Usa HeterogeneousDDPTrainer para clústeres mixtos GPU+CPU.

        En lugar de DDPTrainer (batch uniforme), crea HeterogeneousDDPTrainer
        que normaliza los gradientes por el batch global real (suma de todos los ranks).
        Requiere que el builder tenga distributed=True.
        """
        self._hetero_local_batch_size = max(1, local_batch_size)
        return self

    def with_model_parallel(self, devices, split_block: int | None = None) -> "TrainingSessionBuilder":
        """Split the model across ``devices`` (pipeline model parallelism).

        Builds a ModelParallelViT and a ModelParallelTrainer (single process) so the
        whole decorator stack applies — the model-parallel run gets the same metrics
        as single/DDP. ``devices`` is a list like ``["cuda:0", "cuda:1"]``.
        """
        self._model_parallel = (list(devices), split_block)
        return self

    def with_output_mode(self, mode: str) -> "TrainingSessionBuilder":
        """Override del nombre de modo en las rutas de salida (logs/, plots/, checkpoints/).

        Útil para diferenciar ddp_hetero de ddp en el árbol de artefactos.
        """
        self._output_mode = mode
        return self

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self) -> BaseTrainer:
        """Assemble and return the fully configured trainer stack."""
        from src.training.config_validator import validate_config
        validate_config(self._cfg)

        cfg = self._cfg

        # ── Model ────────────────────────────────────────────────────────────
        model_name = self._model_name or cfg["model"]["name"]
        if self._model_parallel is not None:
            from src.models.model_parallel import build_model_parallel_vit
            mp_devices, mp_split = self._model_parallel
            model = build_model_parallel_vit(
                model_name=model_name,
                num_classes=cfg["model"].get("num_classes", 19),
                pretrained=cfg["model"].get("pretrained", True),
                devices=mp_devices,
                split_block=mp_split,
            )
        else:
            model = build_model(
                model_name=model_name,
                num_classes=cfg["model"].get("num_classes", 19),
                pretrained=cfg["model"].get("pretrained", True),
            )

        # ── Optimizer ────────────────────────────────────────────────────────
        # For model parallelism the LLRD layer structure lives on the wrapped
        # backbone (model.base); the parameters are the same objects either way.
        opt_model = getattr(model, "base", model)
        lr_base = cfg["training"]["lr"]
        weight_decay = cfg["training"].get("weight_decay", 0.05)
        llrd_decay = cfg["training"].get("llrd_decay", 0.0)

        if llrd_decay > 0.0 and opt_model.is_vit:
            optimizer = build_llrd_optimizer(opt_model, lr_base, weight_decay, llrd_decay)
        else:
            if llrd_decay > 0.0 and not opt_model.is_vit:
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
                optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr_min,
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

        # ── Output paths (env/mode/model — needed for checkpoint_dir) ─────────
        env = cfg.get("output", {}).get("env", "local")
        mode = self._output_mode or (
            "model_parallel" if self._model_parallel is not None
            else "ddp" if self._distributed else "single")
        model_slug = model_name.replace("/", "_")

        # ── Base Trainer ──────────────────────────────────────────────────────
        grad_clip = cfg["training"].get("grad_clip", None)
        label_smoothing = cfg["training"].get("label_smoothing", 0.0)
        mixup_alpha = cfg["training"].get("mixup_alpha", 0.0)
        precision = cfg["training"].get("precision", "fp32")
        checkpoint_dir = str(Path(cfg["checkpoint"]["dir"]) / mode / model_slug)

        # ── Loss / criterion (lever against the rare-class macro-F1 ceiling) ───
        # training.loss: 'bce' (default) | 'focal'; training.pos_weight: list | 'auto'
        criterion = self._build_criterion(cfg, model_name)

        if self._model_parallel is not None:
            from src.training.model_parallel_trainer import ModelParallelTrainer
            base = ModelParallelTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=model.output_device,   # logits/labels/loss live on the last stage
                checkpoint_dir=checkpoint_dir,
                criterion=criterion,
                grad_clip=grad_clip,
                label_smoothing=label_smoothing,
                mixup_alpha=mixup_alpha,
                precision=precision,
            )
        elif self._distributed and self._hetero_local_batch_size is not None:
            from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer
            base = HeterogeneousDDPTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self._device,
                checkpoint_dir=checkpoint_dir,
                grad_clip=grad_clip,
                label_smoothing=label_smoothing,
                mixup_alpha=mixup_alpha,
                precision=precision,
                rank=self._rank,
                world_size=self._world_size,
                local_batch_size=self._hetero_local_batch_size,
            )
        elif self._distributed:
            from src.training.ddp_trainer import DDPTrainer
            base = DDPTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self._device,
                checkpoint_dir=checkpoint_dir,
                criterion=criterion,
                grad_clip=grad_clip,
                label_smoothing=label_smoothing,
                mixup_alpha=mixup_alpha,
                precision=precision,
                rank=self._rank,
                world_size=self._world_size,
            )
        else:
            base = Trainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self._device,
                checkpoint_dir=checkpoint_dir,
                criterion=criterion,
                grad_clip=grad_clip,
                label_smoothing=label_smoothing,
                mixup_alpha=mixup_alpha,
                precision=precision,
            )

        # ── 1. @ function decorators on Trainer methods ───────────────────────
        if "energy" in self._fn:
            # Model parallelism spans several GPUs in ONE process → pass the split
            # devices explicitly so the energy is the total across them, not just GPU 0.
            energy_devices = None
            if self._model_parallel is not None:
                _idx = {int(str(d).split(":")[1])
                        for d in self._model_parallel[0] if str(d).startswith("cuda")}
                energy_devices = sorted(_idx) or None   # dedupe: cuda:0,cuda:0 → [0], not [0,0]
            base.train_epoch = measure_energy(base.train_epoch, devices=energy_devices)
            base.eval_epoch = measure_energy(base.eval_epoch, devices=energy_devices)
        if "timing" in self._fn:
            base.train_epoch = timed(base.train_epoch)
            base.eval_epoch = timed(base.eval_epoch)

        # ── Output directories ────────────────────────────────────────────────
        log_dir = Path(f"logs/{env}/{mode}/{model_slug}")
        plot_dir = Path(f"plots/{env}/{mode}/{model_slug}")
        log_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # ── 2. Aspect decorators (inner → outer) ──────────────────────────────
        inner: BaseTrainer = base
        if "hooks" in self._layers:
            inner = LayerHooksDecorator(inner)
        if "batch-monitor" in self._layers:
            # Priority: CLI --batch-log-every > config log_batch_every > default 1
            log_every = (
                self._batch_log_every
                or cfg["training"].get("log_batch_every", 1)
            )
            inner = BatchMonitorDecorator(
                inner,
                log_every=log_every,
                output_dir=str(log_dir),
                timestamp=self._timestamp,
            )
        if "plot" in self._layers:
            inner = PlottingDecorator(
                inner, output_path=str(plot_dir / f"training_{self._timestamp}.png")
            )
        if "confusion" in self._layers:
            inner = ConfusionMatrixDecorator(
                inner, output_dir=str(plot_dir), timestamp=self._timestamp,
                csv_dir=str(log_dir),
            )

        # ── 3. Logger ─────────────────────────────────────────────────────────
        logger: logging.Logger | None = None
        use_deep = self._trace == "deep" or self._inspect is not None

        if self._trace in ("simple", "deep") or use_deep:
            prefix = "train_deep" if use_deep else "train"
            log_file = str(log_dir / f"{prefix}_{self._timestamp}.log")
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
        # Model-selection / early-stopping metric (default: F1 at threshold 0.5).
        select_metric = str(cfg["training"].get("select_by", "f1")).lower()
        epoch_csv = log_dir / f"epoch_metrics_{self._timestamp}.csv" if self._trace != "off" else None

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
                select_metric=select_metric,
            )
        elif self._trace == "off":
            trainer = TracingDecorator(inner, patience=patience,
                                       select_metric=select_metric)
        else:  # simple
            trainer = TracingDecorator(inner, logger=logger, patience=patience,
                                       epoch_csv=epoch_csv, select_metric=select_metric)

        return trainer

    # ── Criterion ──────────────────────────────────────────────────────────────

    def _build_criterion(self, cfg: dict, model_name: str):
        """Build the loss from config: 'bce' (+ optional pos_weight) or 'focal'.

        Returns None to keep the Trainer's default BCEWithLogitsLoss when the
        config asks for plain BCE with no pos_weight (preserves prior behaviour).
        """
        from src.training.losses import build_criterion, pos_weight_from_metadata

        train_cfg = cfg["training"]
        loss_kind = str(train_cfg.get("loss", "bce")).lower()
        pos_weight_cfg = train_cfg.get("pos_weight")

        if self._hetero_local_batch_size is not None and loss_kind == "focal":
            print("  [aviso] loss=focal no soportado en el trainer heterogéneo; "
                  "se usará BCE en ese path.")
            return None

        if loss_kind == "bce" and pos_weight_cfg is None:
            return None  # default path — Trainer creates plain BCEWithLogitsLoss

        pos_weight = None
        if loss_kind == "bce" and pos_weight_cfg is not None:
            if isinstance(pos_weight_cfg, str) and pos_weight_cfg.lower() == "auto":
                pos_weight = pos_weight_from_metadata(
                    cfg["data"]["metadata"], split="train"
                ).to(self._device)
                print(f"  [loss] pos_weight='auto' computado del metadata "
                      f"(min={pos_weight.min():.2f}, max={pos_weight.max():.2f})")
            else:
                pos_weight = torch.tensor(
                    pos_weight_cfg, dtype=torch.float32, device=self._device
                )
        criterion = build_criterion(train_cfg, pos_weight=pos_weight)
        if loss_kind == "focal":
            print(f"  [loss] focal (gamma={train_cfg.get('focal_gamma', 2.0)}, "
                  f"alpha={train_cfg.get('focal_alpha', -1.0)})")
        return criterion

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
