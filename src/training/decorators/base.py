"""Base classes for the OOP Decorator pattern applied to training.

Two abstract levels:
  TrainerDecorator  — pure delegation; override only what you need.
  EpochController   — Template Method for the training loop; subclasses
                      override hook methods instead of reimplementing fit.
                      External observers (``TrainingListener``) subscribe with
                      ``add_listener`` and receive the run lifecycle events.
"""

import time
import warnings

import torch.distributed as dist
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.events import TrainingListener


class TrainerDecorator(BaseTrainer):
    """Base OOP decorator: delegates every method to the wrapped trainer.

    Subclasses override only the methods relevant to their concern.
    __getattr__ transparently exposes model, optimizer, device, criterion, etc.
    """

    def __init__(self, trainer: BaseTrainer):
        self._trainer = trainer

    def __getattr__(self, name: str):
        # Only called when `name` is not found on self — safe to delegate without recursion
        return getattr(self._trainer, name)

    def train_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.train_epoch(loader)

    def eval_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.eval_epoch(loader)

    def save_checkpoint(self, epoch: int, metrics: dict):
        return self._trainer.save_checkpoint(epoch, metrics)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        return self._trainer.fit(train_loader, val_loader, epochs)


class EpochController(TrainerDecorator):
    """Template Method pattern for the training loop.

    Defines the skeleton of training: epoch iteration, best-model tracking,
    checkpointing, ETA, and optional early stopping. Subclasses override the
    _on_* hooks to add logging or other behaviour without touching the loop.

    This is the base for all 'controller' decorators (TracingDecorator,
    DeepTracingDecorator). Only one controller should be active per run —
    it sits at the outermost position of the decorator stack.
    """

    def __init__(self, trainer: BaseTrainer, patience: int | None = None,
                 select_metric: str = "f1"):
        super().__init__(trainer)
        # None = disabled; positive int = stop after N epochs without improvement
        self._patience = patience
        # Metric used to pick the best checkpoint and drive early stopping:
        #   "f1"         → Val F1 at the fixed 0.5 threshold (default; back-compat)
        #   "f1_optimal" → Val F1 at the per-epoch optimal threshold.
        # The latter is the FAIR criterion when the loss shifts the probability
        # scale (e.g. focal loss pushes probabilities down, so 0.5-F1 understates
        # it). Applied identically to every arm, the comparison stays clean.
        #   "auc"        → Val macro ROC AUC: threshold-free, the right choice when
        #                  classes are so rare that F1@0.5 stays near zero.
        self._select_metric = select_metric
        self._listeners: list[TrainingListener] = []

    @property
    def _best_label(self) -> str:
        """Name of the selection metric in log messages (F1 keeps the old wording)."""
        return "AUC" if self._select_metric == "auc" else "F1"

    def _selection_value(self, val_m: dict) -> float:
        """Value used to decide 'is this the best epoch so far?'."""
        if self._select_metric == "f1_optimal":
            return val_m.get("_f1_at_optimal_threshold", val_m["f1"])
        if self._select_metric == "auc":
            return val_m["auc"]
        return val_m["f1"]

    def add_listener(self, listener: TrainingListener) -> None:
        """Subscribe ``listener`` to the run lifecycle (Observer pattern).

        Listeners only observe: they receive the same values as the ``_on_*`` hooks
        after the loop has used them, so they cannot change the training.
        """
        self._listeners.append(listener)

    def _notify(self, event: str, **kwargs) -> None:
        # An observer that fails (full disk, bad path…) must never stop a training run.
        for listener in self._listeners:
            try:
                getattr(listener, event)(**kwargs)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"training listener {type(listener).__name__}.{event} "
                              f"failed: {exc!r}", RuntimeWarning, stacklevel=2)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        # Stack invariant: EpochController (controller) sits outermost; aspect decorators
        # (PlottingDecorator, LayerHooksDecorator) sit between it and the Trainer.
        self._on_fit_start(epochs)
        self._notify("on_fit_start", epochs=epochs)
        progress = {"epochs_run": 0, "best": -1.0}
        try:
            stopped_early = self._run_epochs(train_loader, val_loader, epochs, progress)
        except BaseException as exc:
            self._notify("on_fit_error", error=exc, epochs_run=progress["epochs_run"],
                         best=progress["best"])
            raise
        self._on_fit_end(progress["best"])
        self._notify("on_fit_end", best=progress["best"], epochs_run=progress["epochs_run"],
                     early_stopped=stopped_early)

    def _run_epochs(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int,
                    progress: dict) -> bool:
        """The epoch loop. Updates ``progress`` as it goes (so a failure can still be
        reported) and returns whether early stopping ended it."""
        best_f1 = -1.0   # so the first epoch always saves a baseline checkpoint
        epoch_times: list[float] = []
        epochs_no_improve: int = 0

        for epoch in range(1, epochs + 1):
            self._on_epoch_start(epoch, epochs)
            self._notify("on_epoch_start", epoch=epoch, epochs=epochs)
            t0 = time.time()

            train_m = self.train_epoch(train_loader)
            val_m = self.eval_epoch(val_loader)
            epoch_times.append(time.time() - t0)

            is_best = self._selection_value(val_m) > best_f1
            if is_best:
                best_f1 = self._selection_value(val_m)
                epochs_no_improve = 0
                metrics_clean = {k: v for k, v in val_m.items() if not k.startswith("_")}
                # Keep the threshold tuned on VALIDATION with the checkpoint, so the
                # test evaluation applies it instead of tuning on the test set itself.
                if val_m.get("_optimal_threshold") is not None:
                    metrics_clean["optimal_threshold"] = val_m["_optimal_threshold"]
                self.save_checkpoint(epoch, metrics_clean)
            else:
                epochs_no_improve += 1

            self._on_epoch_end(epoch, epochs, train_m, val_m, best_f1, epoch_times)
            progress["epochs_run"], progress["best"] = epoch, best_f1
            self._notify("on_epoch_end", epoch=epoch, epochs=epochs, train_m=train_m,
                         val_m=val_m, best=best_f1, is_best=is_best,
                         epoch_time_s=epoch_times[-1])

            # Sync all processes at epoch boundary before early-stop decision
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if self._patience is not None and self._should_stop(epochs_no_improve):
                self._on_early_stop(epoch, best_f1)
                self._notify("on_early_stop", epoch=epoch, best=best_f1)
                return True
        return False

    def _should_stop(self, epochs_no_improve: int) -> bool:
        """Early-stopping decision; in DDP rank 0 decides and broadcasts it so all
        ranks agree."""
        if dist.is_available() and dist.is_initialized():
            stop_flag = [epochs_no_improve >= self._patience]
            dist.broadcast_object_list(stop_flag, src=0)
            return stop_flag[0]
        return epochs_no_improve >= self._patience

    # ── Hooks — override in subclasses ──────────────────────────────────────

    def _on_fit_start(self, epochs: int):  # noqa: ARG002
        pass

    def _on_epoch_start(self, epoch: int, epochs: int):  # noqa: ARG002
        pass

    def _on_epoch_end(
        self,
        epoch: int,
        epochs: int,
        train_m: dict,
        val_m: dict,
        best_f1: float,
        epoch_times: list[float],
    ):  # noqa: ARG002
        pass

    def _on_fit_end(self, best_f1: float):  # noqa: ARG002
        pass

    def _on_early_stop(self, epoch: int, best_f1: float):  # noqa: ARG002
        pass
