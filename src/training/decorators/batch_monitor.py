"""BatchMonitorDecorator — batch-level loss monitoring with CSV export.

Reimplements train_epoch to capture running loss at every N batches and
write it to a CSV file. This allows tracking intra-epoch loss dynamics,
which is invisible when only epoch-level metrics are reported.

CSV format: epoch, batch, running_loss
Output: logs/batch_metrics_{timestamp}.csv

Propagates the final train result to inner decorators (e.g. PlottingDecorator)
via the same _propagate_train_result mechanism used by DeepTracingDecorator.
"""

import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.decorators.base import TrainerDecorator
from src.training import metrics as m
from src.training.augmentations import mixup_batch


class BatchMonitorDecorator(TrainerDecorator):
    """Aspect decorator that logs running train loss every N batches to a CSV.

    Activates with --layers batch-monitor. Compatible with --trace off/simple.
    Do NOT combine with --inspect batch-table (both reimplement train_epoch).

    Stack between Trainer and metric reporters, same as other aspect decorators.
    """

    def __init__(
        self,
        trainer,
        log_every: int = 50,
        output_dir: str = "logs",
        timestamp: str = "",
    ):
        super().__init__(trainer)
        self._log_every = log_every
        self._epoch = 0
        csv_name = f"batch_metrics_{timestamp}.csv" if timestamp else "batch_metrics.csv"
        self._csv_path = Path(output_dir) / csv_name
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w") as f:
            f.write("epoch,batch,n_batches,running_loss\n")

    def train_epoch(self, loader: DataLoader) -> dict:
        model = self._trainer.model
        optimizer = self._trainer.optimizer
        criterion = self._trainer.criterion
        device = self._trainer.device

        model.train()
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        start = time.time()
        self._epoch += 1

        label_smoothing = getattr(self._trainer, "label_smoothing", 0.0)
        mixup_alpha = getattr(self._trainer, "mixup_alpha", 0.0)

        for batch_idx, (images, labels) in enumerate(loader, 1):
            images, labels = images.to(device), labels.to(device)

            if mixup_alpha > 0.0 and random.random() < 0.5:
                images, labels = mixup_batch(images, labels, mixup_alpha)
            if label_smoothing > 0.0:
                labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            grad_clip = getattr(self._trainer, "grad_clip", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                hard_labels = (labels > 0.5).long()
                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.append(preds.cpu())
                all_labels.append(hard_labels.cpu())

            if batch_idx % self._log_every == 0:
                running_loss = total_loss / batch_idx
                with open(self._csv_path, "a") as f:
                    f.write(f"{self._epoch},{batch_idx},{len(loader)},{running_loss:.6f}\n")

        scheduler = getattr(self._trainer, "scheduler", None)
        if scheduler:
            scheduler.step()

        all_preds_t = torch.cat(all_preds)
        all_labels_t = torch.cat(all_labels)
        result = {
            "loss": total_loss / len(loader),
            "f1": m.f1_score(all_preds_t, all_labels_t),
            "accuracy": m.accuracy(all_preds_t, all_labels_t),
            "time": time.time() - start,
        }
        self._propagate_train_result(result)
        return result

    def _propagate_train_result(self, result: dict):
        """Notify inner decorators of train metrics when we own the training loop."""
        inner = self._trainer
        while hasattr(inner, "_trainer"):
            if hasattr(inner, "_record_train_result"):
                inner._record_train_result(result)
            inner = inner._trainer
