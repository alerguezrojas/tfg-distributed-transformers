"""TracingDecorator — epoch-level logging controller.

Works as console logger (logger=None) or structured file logger (logger=Logger).
Extends EpochController: only overrides the _on_* hooks, never the fit loop.
"""

import logging

from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.metrics import eta_str
from src.training.decorators.base import EpochController


class TracingDecorator(EpochController):
    """Logs train/val metrics after each epoch to console and/or a file.

    Pass logger=None for console-only output (--trace off).
    Pass a Logger instance for structured file output (--trace simple).

    Stack aspect decorators (PlottingDecorator, LayerHooksDecorator) between
    this controller and the Trainer — they are invoked transparently via the
    train_epoch / eval_epoch delegation chain.

    Example:
        trainer = TracingDecorator(
            PlottingDecorator(Trainer(...), output_path="plots/run.png"),
            logger=setup_logger("trainer", log_file="logs/train.log"),
        )
        trainer.fit(train_loader, val_loader, epochs=30)
    """

    def __init__(self, trainer: BaseTrainer, logger: logging.Logger | None = None):
        super().__init__(trainer)
        self._logger = logger

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _emit(self, msg: str):
        if self._logger:
            self._logger.info(msg)
        else:
            print(msg)

    # ── EpochController hooks ────────────────────────────────────────────────

    def _on_fit_start(self, epochs: int):
        self._emit(f"Iniciando entrenamiento — {epochs} epochs")

    def _on_epoch_start(self, epoch: int, epochs: int):
        self._emit(f"[Epoch {epoch:03d}/{epochs}] Entrenando...")

    def _on_epoch_end(self, epoch, epochs, train_m, val_m, best_f1, epoch_times):
        self._emit(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_m['loss']:.4f}  train_f1={train_m['f1']:.4f}  train_acc={train_m['accuracy']:.4f} | "
            f"val_loss={val_m['loss']:.4f}  val_f1={val_m['f1']:.4f}  best={best_f1:.4f}  val_acc={val_m['accuracy']:.4f} | "
            f"time={train_m['time']:.0f}s  ETA={eta_str(epoch_times, epoch, epochs)}"
        )

    def _on_fit_end(self, best_f1: float):
        self._emit(f"Entrenamiento completado — mejor Val F1: {best_f1:.4f}")

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)
        self._emit(f"[Epoch {epoch:03d}] Checkpoint guardado")
