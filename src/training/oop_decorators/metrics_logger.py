import time

from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training.metrics import eta_str
from src.training.oop_decorators.base import TrainerDecorator


class MetricsLoggerDecorator(TrainerDecorator):
    """Prints train/val metrics to stdout after each epoch with best F1 and ETA."""

    def train_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.train_epoch(loader)

    def eval_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.eval_epoch(loader)

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)
        print(f"  Checkpoint guardado (epoch {epoch:03d})")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        best_f1 = 0.0
        epoch_times: list[float] = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch(train_loader)
            val_m = self.eval_epoch(val_loader)
            epoch_times.append(time.time() - t0)

            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                self.save_checkpoint(epoch, val_m)

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_m['loss']:.4f}  train_f1={train_m['f1']:.4f}  train_acc={train_m['accuracy']:.4f} | "
                f"val_loss={val_m['loss']:.4f}  val_f1={val_m['f1']:.4f}  best={best_f1:.4f}  val_acc={val_m['accuracy']:.4f} | "
                f"time={train_m['time']:.0f}s  ETA={eta_str(epoch_times, epoch, epochs)}"
            )
