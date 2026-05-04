import logging
import time

from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.metrics import eta_str
from src.training.oop_decorators.base import TrainerDecorator


class TracingDecorator(TrainerDecorator):
    """Structured epoch-level logging with timestamps to console and file.

    Suitable for production training runs: no per-batch overhead,
    full metrics emitted as INFO log lines after each epoch.
    """

    def __init__(self, trainer: BaseTrainer, logger: logging.Logger):
        super().__init__(trainer)
        self._logger = logger

    def train_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.train_epoch(loader)

    def eval_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.eval_epoch(loader)

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)
        self._logger.info(f"[Epoch {epoch:03d}] Checkpoint guardado")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self._logger.info(f"Iniciando entrenamiento — {epochs} epochs")
        best_f1 = 0.0
        epoch_times: list[float] = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            self._logger.info(f"[Epoch {epoch:03d}/{epochs}] Entrenando...")
            train_m = self.train_epoch(train_loader)

            self._logger.info(f"[Epoch {epoch:03d}/{epochs}] Evaluando...")
            val_m = self.eval_epoch(val_loader)
            epoch_times.append(time.time() - t0)

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                self.save_checkpoint(epoch, val_m)

            self._logger.info(
                f"[Epoch {epoch:03d}/{epochs}] "
                f"train_loss={train_m['loss']:.4f}  train_f1={train_m['f1']:.4f}  train_acc={train_m['accuracy']:.4f} | "
                f"val_loss={val_m['loss']:.4f}  val_f1={val_m['f1']:.4f}  best={best_f1:.4f}  val_acc={val_m['accuracy']:.4f} | "
                f"time={train_m['time']:.0f}s  ETA={eta_str(epoch_times, epoch, epochs)}"
            )

        self._logger.info(f"Entrenamiento completado — mejor Val F1: {best_f1:.4f}")
