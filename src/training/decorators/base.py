"""Base classes for the OOP Decorator pattern applied to training.

Two abstract levels:
  TrainerDecorator  — pure delegation; override only what you need.
  EpochController   — Template Method for the training loop; subclasses
                      override hook methods instead of reimplementing fit.
"""

import time

from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer


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

    def __init__(self, trainer: BaseTrainer, patience: int | None = None):
        super().__init__(trainer)
        # None = disabled; positive int = stop after N epochs without improvement
        self._patience = patience

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        # Stack invariant: EpochController (controller) sits outermost; aspect decorators
        # (PlottingDecorator, LayerHooksDecorator) sit between it and the Trainer.
        best_f1 = 0.0
        epoch_times: list[float] = []
        epochs_no_improve: int = 0

        self._on_fit_start(epochs)

        for epoch in range(1, epochs + 1):
            self._on_epoch_start(epoch, epochs)
            t0 = time.time()

            train_m = self.train_epoch(train_loader)
            val_m = self.eval_epoch(val_loader)
            epoch_times.append(time.time() - t0)

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                epochs_no_improve = 0
                self.save_checkpoint(epoch, val_m)
            else:
                epochs_no_improve += 1

            self._on_epoch_end(epoch, epochs, train_m, val_m, best_f1, epoch_times)

            if self._patience is not None and epochs_no_improve >= self._patience:
                self._on_early_stop(epoch, best_f1)
                break

        self._on_fit_end(best_f1)

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
