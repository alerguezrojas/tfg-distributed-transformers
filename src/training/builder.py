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
import warnings
from datetime import datetime
from pathlib import Path

import torch

from src.training.base_trainer import BaseTrainer
from src.training.trainer import Trainer
from src.training.logger_setup import setup_logger
from src.training.fn_decorators import timed, measure_energy
from src.training.events import JsonlEventSink, environment_fingerprint
from src.training.alerts import AlertMonitor, alerts_config
from src.training.monitor import LayerStatsMonitor, PredictionMonitor, monitor_config
from src.training.system_sampler import SystemSampler, visible_gpu_uuids
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
from src.data.registry import DEFAULT_DATASET, dataset_spec_from_config
from src.data.spec import DatasetSpec


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
        self._monitor: bool = False                   # --monitor (or monitor.enabled)

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

    def with_monitor(self, enabled: bool = True) -> "TrainingSessionBuilder":
        """Enable the prediction and layer monitors (same as ``monitor.enabled``)."""
        self._monitor = enabled
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

        # ── Dataset & task ───────────────────────────────────────────────────
        # The dataset spec fixes the classes, input channels and problem type;
        # the task decides loss, predictions and metrics for every trainer.
        spec = dataset_spec_from_config(cfg)
        task = spec.make_task()

        # ── Model ────────────────────────────────────────────────────────────
        model_name = self._model_name or cfg["model"]["name"]
        dropout = cfg["model"].get("dropout", 0.1)
        if self._model_parallel is not None:
            from src.models.model_parallel import build_model_parallel_vit
            mp_devices, mp_split = self._model_parallel
            model = build_model_parallel_vit(
                model_name=model_name,
                num_classes=spec.num_classes,
                pretrained=cfg["model"].get("pretrained", True),
                devices=mp_devices,
                split_block=mp_split,
                in_chans=spec.in_channels,
                dropout=dropout,
            )
        else:
            model = build_model(
                model_name=model_name,
                num_classes=spec.num_classes,
                pretrained=cfg["model"].get("pretrained", True),
                in_chans=spec.in_channels,
                dropout=dropout,
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

        # ── Output paths (env/[dataset/]mode/model — needed for checkpoint_dir) ─
        env = cfg.get("output", {}).get("env", "local")
        mode = self._output_mode or (
            "model_parallel" if self._model_parallel is not None
            else "ddp" if self._distributed else "single")
        run_subdir = self.run_subdir(spec, mode, model_name)

        # ── Base Trainer ──────────────────────────────────────────────────────
        grad_clip = cfg["training"].get("grad_clip", None)
        label_smoothing = cfg["training"].get("label_smoothing", 0.0)
        mixup_alpha = cfg["training"].get("mixup_alpha", 0.0)
        precision = cfg["training"].get("precision", "fp32")
        checkpoint_dir = str(Path(cfg["checkpoint"]["dir"]) / run_subdir)

        # ── Loss / criterion (lever against the rare-class macro-F1 ceiling) ───
        # training.loss: task default ('bce' multi-label, 'ce' multi-class) | 'focal';
        # training.pos_weight: list | 'auto' (sigmoid tasks)
        criterion = self._build_criterion(cfg, spec)

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
                task=task,
            )
        elif self._distributed and self._hetero_local_batch_size is not None:
            from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer
            base = HeterogeneousDDPTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self._device,
                checkpoint_dir=checkpoint_dir,
                criterion=criterion,      # its pos_weight carries into the summed loss
                grad_clip=grad_clip,
                label_smoothing=label_smoothing,
                mixup_alpha=mixup_alpha,
                precision=precision,
                task=task,
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
                task=task,
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
                task=task,
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
        log_dir = Path("logs") / env / run_subdir
        plot_dir = Path("plots") / env / run_subdir
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
                csv_dir=str(log_dir), class_names=spec.classes,
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

        # ── 6. Event stream (rank 0) — docs/eventos.md ────────────────────────
        if self._trace != "off" and self._rank == 0:
            sink = JsonlEventSink(
                log_dir / f"events_{self._timestamp}.jsonl",
                run_info={
                    "run_id": self._timestamp,
                    "dataset": spec.name,
                    "task": task.name,
                    "classes": list(spec.classes),
                    "model": model_name,
                    "mode": mode,
                    "world_size": self._world_size,
                    "select_metric": select_metric,
                    "config": cfg,
                    "env": environment_fingerprint(),
                },
                log_every=self._batch_log_every or cfg["training"].get("log_batch_every", 1),
            )
            # System sampler (GPU/CPU/RAM every N s). Added before the sink so it stops
            # before `run_end` is written. 0 disables it; independent of the monitor.
            system_interval = (cfg.get("monitor") or {}).get("system_interval_s", 5)
            if system_interval and system_interval > 0:
                uuids = visible_gpu_uuids() if self._device.type == "cuda" else None
                trainer.add_listener(SystemSampler(sink, interval_s=system_interval,
                                                   gpu_uuids=uuids))
            trainer.add_listener(sink)
            if "batch-monitor" in self._layers:
                # Same cadence as the batch CSV (the monitor already makes the
                # Trainer compute the per-batch metrics).
                base.register_batch_hook(sink.on_batch)
            # Alerts first: their batch hook must run before the layer monitor's, so a
            # batch's loss is known when its `layers` event is evaluated.
            self._attach_alerts(trainer, base, sink, cfg)
            self._attach_monitors(trainer, base, sink, cfg, spec, task, log_dir)
        elif self._trace == "off" and (self._monitor or monitor_config(cfg)["enabled"]):
            warnings.warn("the prediction monitor writes to the event stream, which "
                          "--trace off disables: the monitor is not active",
                          RuntimeWarning, stacklevel=2)

        return trainer

    # ── Inspection (docs/inspeccion.md) ───────────────────────────────────────

    def _attach_monitors(self, trainer, base, sink, cfg, spec, task, log_dir) -> None:
        """Prediction and layer monitors (``monitor.enabled`` or ``--monitor``)."""
        mon = monitor_config(cfg)
        if not (mon["enabled"] or self._monitor):
            return
        # Note: with layer statistics the Trainer has a batch hook, so it computes the
        # per-batch metrics dict every batch (small CPU work; no number changes).
        data_cfg = cfg["data"]
        trainer.add_listener(PredictionMonitor(
            base, spec.dataset(data_cfg, "val"), task, spec.classes, sink, log_dir,
            self._timestamp, mon, batch_size=cfg["training"]["batch_size"],
            num_workers=min(int(data_cfg.get("num_workers", 0)), 4)))
        if mon["layers"]:
            layers = LayerStatsMonitor(
                base, sink,
                log_every=self._batch_log_every or cfg["training"].get("log_batch_every", 50))
            trainer.add_listener(layers)
            base.register_batch_hook(layers.on_batch)

    def _attach_alerts(self, trainer, base, sink, cfg) -> None:
        """Alert rules over the event stream (``monitor.alerts``, on by default).

        Added after the sink so each alert follows the ``epoch_end`` it comes from. It
        listens to the sink for ``system`` / ``layers`` events, and takes the per-batch NaN
        and gradient-norm checks only if the run has batch hooks anyway (batch monitor or
        layer statistics): it never forces per-batch work, otherwise the epoch loss is
        checked."""
        acfg = alerts_config(cfg)
        if not acfg["enabled"]:
            return
        alerts = AlertMonitor(sink, acfg, grad_norm_source=lambda: base.last_grad_norm)
        trainer.add_listener(alerts)
        sink.subscribe(alerts.on_event)
        mon = monitor_config(cfg)
        layer_stats = (mon["enabled"] or self._monitor) and mon["layers"]
        if "batch-monitor" in self._layers or layer_stats:
            base.register_batch_hook(alerts.on_batch)

    # ── Output layout ─────────────────────────────────────────────────────────

    @staticmethod
    def run_subdir(spec: DatasetSpec, mode: str, model_name: str) -> Path:
        """``mode/model`` for BigEarthNet (the historical layout every existing run
        uses) and ``dataset/mode/model`` for any other dataset, so runs of different
        datasets never share a folder."""
        model_slug = model_name.replace("/", "_")
        if spec.name == DEFAULT_DATASET:
            return Path(mode) / model_slug
        return Path(spec.name) / mode / model_slug

    # ── Criterion ──────────────────────────────────────────────────────────────

    def _build_criterion(self, cfg: dict, spec: DatasetSpec):
        """Loss for the dataset's task from ``training.loss`` (+ ``pos_weight``).

        ``pos_weight: auto`` derives the weights from the class frequencies of the
        training split, which the dataset spec provides.
        """
        from src.training.losses import build_criterion, pos_weight_from_counts

        train_cfg = cfg["training"]
        loss_kind = str(train_cfg.get("loss") or "").lower()
        pos_weight_cfg = train_cfg.get("pos_weight")

        if self._hetero_local_batch_size is not None and loss_kind == "focal":
            print("  [aviso] loss=focal no soportado en el trainer heterogéneo; "
                  "se usará BCE en ese path.")
            train_cfg = {**train_cfg, "loss": None}

        pos_weight = None
        if pos_weight_cfg is not None and loss_kind != "focal":
            if isinstance(pos_weight_cfg, str) and pos_weight_cfg.lower() == "auto":
                if spec.class_counts is None:
                    raise ValueError(f"pos_weight: auto is not available for dataset "
                                     f"{spec.name!r} (it exposes no class counts)")
                pos_weight = pos_weight_from_counts(
                    *spec.class_counts(cfg["data"], "train")
                ).to(self._device)
                print(f"  [loss] pos_weight='auto' computado del dataset "
                      f"(min={pos_weight.min():.2f}, max={pos_weight.max():.2f})")
            else:
                pos_weight = torch.tensor(
                    pos_weight_cfg, dtype=torch.float32, device=self._device
                )
        criterion = build_criterion(train_cfg, pos_weight=pos_weight, task=spec.task)
        if loss_kind == "focal" and self._hetero_local_batch_size is None:
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
