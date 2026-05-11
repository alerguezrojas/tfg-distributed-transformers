"""Single-GPU trainer for BigEarthNet multi-label classification."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer
from src.training import metrics as m


class Trainer(BaseTrainer):
    """Single-GPU trainer for BigEarthNet classification.

    Pure training logic — no logging or printing.
    Wrap with an OOP decorator or apply Python @ decorators to its methods.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        grad_clip: float | None = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = grad_clip
        # Model returns raw logits (no sigmoid) — BCEWithLogitsLoss applies sigmoid
        # internally. Switching to BCELoss would require adding sigmoid to the model.
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        start = time.time()

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                preds = torch.sigmoid(logits) > 0.5
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        if self.scheduler:
            self.scheduler.step()

        all_preds_t = torch.cat(all_preds)
        all_labels_t = torch.cat(all_labels)

        return {
            "loss": total_loss / len(loader),
            "f1": m.f1_score(all_preds_t, all_labels_t),
            "accuracy": m.accuracy(all_preds_t, all_labels_t),
            "time": time.time() - start,
        }

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits) > 0.5
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds_t = torch.cat(all_preds)
        all_labels_t = torch.cat(all_labels)

        return {
            "loss": total_loss / len(loader),
            "f1": m.f1_score(all_preds_t, all_labels_t),
            "accuracy": m.accuracy(all_preds_t, all_labels_t),
            "precision": m.precision(all_preds_t, all_labels_t),
            "recall": m.recall(all_preds_t, all_labels_t),
            # Tensors for per-class analysis (consumed by ConfusionMatrixDecorator if active)
            "_preds": all_preds_t,
            "_labels": all_labels_t,
        }

    def save_checkpoint(self, epoch: int, metrics: dict):
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        best_f1 = 0.0
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self.save_checkpoint(epoch, val_metrics)
