import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """Single-GPU trainer for BigEarthNet classification.

    Pure training logic — no logging or printing.
    Wrap with a decorator to add console output or tracing.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
            "f1": self._f1_score(all_preds_t, all_labels_t),
            "accuracy": self._accuracy(all_preds_t, all_labels_t),
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
            "f1": self._f1_score(all_preds_t, all_labels_t),
            "accuracy": self._accuracy(all_preds_t, all_labels_t),
            "precision": self._precision(all_preds_t, all_labels_t),
            "recall": self._recall(all_preds_t, all_labels_t),
        }

    def _f1_score(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """Macro-averaged F1 for multi-label classification."""
        tp = (preds & labels.bool()).sum(dim=0).float()
        fp = (preds & ~labels.bool()).sum(dim=0).float()
        fn = (~preds & labels.bool()).sum(dim=0).float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.mean().item()

    def _precision(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """Macro-averaged precision for multi-label classification."""
        tp = (preds & labels.bool()).sum(dim=0).float()
        fp = (preds & ~labels.bool()).sum(dim=0).float()
        return (tp / (tp + fp + 1e-8)).mean().item()

    def _recall(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """Macro-averaged recall for multi-label classification."""
        tp = (preds & labels.bool()).sum(dim=0).float()
        fn = (~preds & labels.bool()).sum(dim=0).float()
        return (tp / (tp + fn + 1e-8)).mean().item()

    def _accuracy(self, preds: torch.Tensor, labels: torch.Tensor) -> float:
        """Sample-averaged accuracy for multi-label classification."""
        correct = (preds == labels.bool()).float().mean(dim=1)
        return correct.mean().item()

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
