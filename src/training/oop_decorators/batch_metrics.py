import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.base_trainer import BaseTrainer
from src.training.oop_decorators.base import TrainerDecorator


class BatchMetricsDecorator(TrainerDecorator):
    """Shows a real-time tqdm progress bar with loss updated per batch.

    Kept for didactic value: illustrates a white-box decorator that
    intercepts the training loop at batch level.
    """

    def __init__(self, trainer: BaseTrainer, log_every: int = 50):
        super().__init__(trainer)
        self.log_every = log_every

    def train_epoch(self, loader: DataLoader) -> dict:
        model = self._trainer.model
        optimizer = self._trainer.optimizer
        criterion = self._trainer.criterion
        device = self._trainer.device
        scheduler = self._trainer.scheduler

        model.train()
        total_loss = 0.0
        start = time.time()

        progress = tqdm(loader, desc="  Train", leave=False, unit="batch")
        for batch_idx, (images, labels) in enumerate(progress, 1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{total_loss / batch_idx:.4f}")

        if scheduler:
            scheduler.step()

        return {
            "loss": total_loss / len(loader),
            "f1": 0.0,
            "accuracy": 0.0,
            "time": time.time() - start,
        }

    def eval_epoch(self, loader: DataLoader) -> dict:
        return self._trainer.eval_epoch(loader)

    def save_checkpoint(self, epoch: int, metrics: dict):
        self._trainer.save_checkpoint(epoch, metrics)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        self._trainer.fit(train_loader, val_loader, epochs)
