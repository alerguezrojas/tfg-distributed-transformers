from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer


class MetricsLoggerDecorator(BaseTrainer):
    """Decorator that adds console metrics logging to any BaseTrainer.

    Wraps a trainer and prints epoch-level metrics after each epoch,
    without modifying the training algorithm itself.

    Usage:
        trainer = MetricsLoggerDecorator(Trainer(model, optimizer, ...))
        trainer.fit(train_loader, val_loader, epochs=30)
    """

    def __init__(self, trainer: BaseTrainer):
        self._trainer = trainer

    def train_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.train_epoch(loader)

    def eval_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.eval_epoch(loader)

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)
        print(f"  Checkpoint guardado (epoch {epoch:03d})")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        best_f1 = 0.0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)

            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_metrics['loss']:.4f} | "
                f"Val loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Time: {train_metrics['time']:.1f}s"
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self.save_checkpoint(epoch, val_metrics)
