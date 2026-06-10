"""Single-GPU trainer for BigEarthNet multi-label classification."""

import contextlib
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import precision as precision_mod
from src.training.base_trainer import BaseTrainer
from src.training import metrics as m
from src.training.augmentations import mixup_batch

_THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


class Trainer(BaseTrainer):
    """Single-GPU trainer for BigEarthNet classification.

    Pure training logic — no logging or printing.
    Wrap with an OOP decorator or apply Python @ decorators to its methods.

    Batch hooks
    -----------
    Register callables via ``register_batch_hook(fn)`` to receive a
    notification after every training batch without reimplementing the loop:

        fn(epoch, batch_idx, n_batches, metrics: dict)

    where ``metrics`` contains:
        running_loss — average loss over all batches so far in the epoch
        batch_loss   — loss of this specific batch (instantaneous)
        lr           — current learning rate (first param group)
        batch_f1     — macro F1 of this batch (indicative; post-mixup labels)
        batch_acc    — sample accuracy of this batch
        batch_prec   — macro precision of this batch

    BatchMonitorDecorator uses this mechanism instead of duplicating train_epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        criterion: nn.Module | None = None,
        grad_clip: float | None = None,
        label_smoothing: float = 0.0,
        mixup_alpha: float = 0.0,
        precision: str = "fp32",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()
        self.grad_clip = grad_clip
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self._current_epoch: int = 0
        self._batch_hooks: list = []

        # ── Numeric precision (the practical Tensor-core switch) ───────────────
        # fp32 -> CUDA cores; tf32/amp/bf16 -> Tensor cores. autocast/GradScaler
        # only apply on CUDA; on CPU we always run fp32.
        self.precision = precision if device.type == "cuda" else "fp32"
        precision_mod.apply_backend_flags(self.precision)
        self._amp_dtype = precision_mod.autocast_dtype(self.precision)
        self._use_amp = self._amp_dtype is not None and device.type == "cuda"
        _scaler_on = precision_mod.needs_scaler(self.precision) and device.type == "cuda"
        try:                                   # torch >= 2.3
            self._scaler = torch.amp.GradScaler("cuda", enabled=_scaler_on)
        except (AttributeError, TypeError):    # older torch
            self._scaler = torch.cuda.amp.GradScaler(enabled=_scaler_on)

    def _autocast(self):
        """Autocast context for the forward pass (no-op for fp32/tf32)."""
        if self._use_amp:
            return torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    def register_batch_hook(self, fn) -> None:
        """Register a callable called after every training batch."""
        self._batch_hooks.append(fn)

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        self._current_epoch += 1
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        start = time.time()
        n_batches = len(loader)

        for batch_idx, (images, labels) in enumerate(loader, 1):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.mixup_alpha > 0.0 and random.random() < 0.5:
                images, labels = mixup_batch(images, labels, self.mixup_alpha)
            if self.label_smoothing > 0.0:
                labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

            self.optimizer.zero_grad()
            with self._autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            self._scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()

            total_loss += loss.item()
            with torch.no_grad():
                hard_labels = (labels > 0.5).long()
                preds = (torch.sigmoid(logits) > 0.5).long()
                preds_cpu = preds.cpu()
                hard_labels_cpu = hard_labels.cpu()
                all_preds.append(preds_cpu)
                all_labels.append(hard_labels_cpu)

            if self._batch_hooks:
                batch_metrics = {
                    "running_loss": total_loss / batch_idx,
                    "batch_loss": loss.item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "batch_f1": m.f1_score(preds_cpu, hard_labels_cpu),
                    "batch_acc": m.accuracy(preds_cpu, hard_labels_cpu),
                    "batch_prec": m.precision(preds_cpu, hard_labels_cpu),
                }
                for hook in self._batch_hooks:
                    hook(self._current_epoch, batch_idx, n_batches, batch_metrics)

        if self.scheduler:
            self.scheduler.step()

        all_preds_t = torch.cat(all_preds)
        all_labels_t = torch.cat(all_labels)

        return {
            "loss": total_loss / n_batches,
            "f1": m.f1_score(all_preds_t, all_labels_t),
            "accuracy": m.accuracy(all_preds_t, all_labels_t),
            "time": time.time() - start,
        }

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with self._autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            total_loss += loss.item()

            all_probs.append(torch.sigmoid(logits.float()).cpu())
            all_labels.append(labels.cpu())

        all_probs_t = torch.cat(all_probs)
        all_labels_t = torch.cat(all_labels)
        all_preds_t = (all_probs_t > 0.5).long()

        f1_base = m.f1_score(all_preds_t, all_labels_t)

        # Threshold grid search on validation set to find optimal F1 threshold
        best_thresh, best_f1_thresh = 0.5, f1_base
        for t in _THRESHOLD_GRID:
            preds_t = (all_probs_t > t).long()
            f1_t = m.f1_score(preds_t, all_labels_t)
            if f1_t > best_f1_thresh:
                best_thresh, best_f1_thresh = t, f1_t

        return {
            "loss": total_loss / len(loader),
            "f1": f1_base,                          # at threshold=0.5 (used for checkpoint criterion)
            "accuracy": m.accuracy(all_preds_t, all_labels_t),
            "precision": m.precision(all_preds_t, all_labels_t),
            "recall": m.recall(all_preds_t, all_labels_t),
            "_optimal_threshold": best_thresh,
            "_f1_at_optimal_threshold": best_f1_thresh,
            # Raw tensors for ConfusionMatrixDecorator (consumed by decorator, stripped before checkpoint)
            "_preds": all_preds_t,
            "_labels": all_labels_t,
        }

    def save_checkpoint(self, epoch: int, metrics: dict):
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path) -> dict:
        """Restore model, optimizer and scheduler from a checkpoint file.

        Returns the checkpoint dict (contains 'epoch' and 'metrics').
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        resumed_epoch = ckpt.get("epoch", 0)
        self._current_epoch = resumed_epoch
        return ckpt

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        best_f1 = 0.0
        for epoch in range(1, epochs + 1):
            self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self.save_checkpoint(epoch, val_metrics)
