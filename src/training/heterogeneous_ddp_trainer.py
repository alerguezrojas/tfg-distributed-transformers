"""HeterogeneousDDPTrainer — DDP trainer for mixed-hardware clusters.

Extends DDPTrainer to handle ranks with different batch sizes correctly.

Problem with standard DDP + unequal batch sizes
-------------------------------------------------
Standard DDP averages gradients across all ranks with equal weight.  If rank 0
has batch_size=64 and rank 1 has batch_size=4, averaging their losses equally
over-weights rank 1's noisy gradient (1/2 weight instead of the correct 4/68).

Correct solution: weighted gradient normalization
-------------------------------------------------
Each rank scales its loss by:

    scale_r = (local_batch_size × world_size) / global_batch_size

where  global_batch_size = Σ local_batch_size_i  (sum over all ranks, gathered
via all_reduce at the start of each batch).

After DDP's built-in AVG all_reduce of gradients, the result equals the
gradient that would be computed on the concatenated global mini-batch.

Equivalently, instead of:
    loss = criterion(logits, labels)          # mean over local batch
we compute:
    loss = criterion_sum(logits, labels)       # sum over local batch
    loss = loss / global_batch_size            # divide by global total

This is what this trainer does inside train_epoch.

Launch with torchrun (gloo backend, mixed GPU+CPU):
    # Node 0 — verode21 (V100 GPU):
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \\
      --master_addr=verode21 --master_port=29500 \\
      scripts/train_heterogeneous_ddp.py \\
      --config configs/train_heterogeneous_ddp.yaml --trace simple

    # Node 1 — verode16 (CPU):
    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \\
      --master_addr=verode21 --master_port=29500 \\
      scripts/train_heterogeneous_ddp.py \\
      --config configs/train_heterogeneous_ddp.yaml --trace simple
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.training.ddp_trainer import DDPTrainer
from src.training import metrics as m


class HeterogeneousDDPTrainer(DDPTrainer):
    """DDP trainer that handles heterogeneous batch sizes across ranks.

    Parameters
    ----------
    local_batch_size: Number of samples this rank processes per step.
                      Needed for correct gradient weighting.
    All other parameters: inherited from DDPTrainer / Trainer.
    """

    def __init__(self, *args, local_batch_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_batch_size = local_batch_size
        # Replace criterion with sum reduction for manual normalization
        self._criterion_sum = nn.BCEWithLogitsLoss(reduction="sum")

    # ── Override train_epoch for weighted gradient normalization ──────────────

    def train_epoch(self, loader: DataLoader) -> dict:
        self._epoch += 1
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(self._epoch)

        self.model.train()
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            step_bs = labels.shape[0]

            # Gather global batch size across all ranks
            bs_tensor = torch.tensor(float(step_bs), device=self.device)
            dist.all_reduce(bs_tensor, op=dist.ReduceOp.SUM)
            global_bs = bs_tensor.item()

            self.optimizer.zero_grad()

            # Apply mixup if configured
            if self.mixup_alpha > 0 and torch.rand(1).item() < 0.5:
                from src.training.augmentations import mixup_batch
                images, labels_a, labels_b, lam = mixup_batch(images, labels, self.mixup_alpha)
                logits = self.model(images)
                loss_sum = (
                    self._criterion_sum(logits, _smooth(labels_a, self.label_smoothing)) * lam
                    + self._criterion_sum(logits, _smooth(labels_b, self.label_smoothing)) * (1 - lam)
                )
            else:
                if self.label_smoothing > 0:
                    labels = _smooth(labels, self.label_smoothing)
                logits = self.model(images)
                loss_sum = self._criterion_sum(logits, labels)

            # Normalize by global batch size (correct weighted gradient)
            loss = loss_sum / global_bs

            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            # Track (use mean loss for reporting)
            total_loss += (loss_sum.detach().item() / step_bs)

            with torch.no_grad():
                preds = (torch.sigmoid(logits.detach()) >= 0.5).cpu()
                all_preds.append(preds)
                # Store original labels (before smoothing) for metrics
                all_labels.append((labels.detach() >= 0.5).cpu())

        if self.scheduler is not None:
            self.scheduler.step()

        n_batches = len(loader)
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        preds_t = torch.cat(all_preds)
        labels_t = torch.cat(all_labels)

        return {
            "loss": avg_loss,
            "f1": m.f1_score(preds_t, labels_t),
            "accuracy": m.accuracy(preds_t, labels_t),
            "_preds": preds_t,
            "_labels": labels_t,
        }


def _smooth(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    return labels * (1 - smoothing) + smoothing / 2
