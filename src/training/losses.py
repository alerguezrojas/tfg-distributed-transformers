"""Loss functions for multi-label classification — levers against the macro-F1
ceiling caused by class imbalance (rare CORINE classes the model never predicts).

Two interchangeable options, both compatible with soft targets (mixup) since they
operate element-wise on targets in [0, 1]:

  • ``BCEWithLogitsLoss(pos_weight=…)`` — re-weights the positive term of each
    class by neg/pos, so rare positives count more. ``pos_weight='auto'`` in the
    config computes the weights from the training-split class frequencies.
  • ``FocalLoss`` (Lin et al. 2017, multi-label variant) — down-weights easy
    examples via ``(1-p_t)^gamma`` so the loss focuses on the hard, rare ones.

``build_criterion`` is a pure factory selected by ``training.loss`` in the config;
which losses are valid depends on the task (see :mod:`src.training.tasks`). The
class frequencies behind ``pos_weight: auto`` come from the dataset spec
(``DatasetSpec.class_counts``), so this module stays free of any file I/O.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.tasks import get_task


class FocalLoss(nn.Module):
    """Multi-label focal loss on raw logits (sigmoid-based, no softmax).

    loss = α_t · (1 − p_t)^γ · BCE(logits, target)

    where p_t is the model's probability of the actual target for each
    (sample, class) element. Works with soft targets in [0, 1].

    Args:
        gamma:  focusing parameter; 0 reduces to plain BCE. Typical: 2.0.
        alpha:  class-balancing weight for the positive term in [0, 1];
                use a negative value (default) to disable α-balancing.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = -1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)   # prob of the actual target
        loss = ((1.0 - p_t) ** self.gamma) * bce
        if self.alpha >= 0.0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def pos_weight_from_counts(pos_counts: torch.Tensor, n_samples: int,
                           clamp_max: float = 100.0) -> torch.Tensor:
    """BCE pos_weight per class = #negatives / #positives, clamped.

    A class never seen positive gets ``clamp_max`` (instead of +inf).
    """
    pos = pos_counts.float()
    neg = (n_samples - pos).clamp(min=0.0)
    weight = neg / pos.clamp(min=1.0)
    weight = torch.where(pos > 0, weight, torch.full_like(weight, clamp_max))
    return weight.clamp(max=clamp_max)


def build_criterion(train_cfg: dict, pos_weight: torch.Tensor | None = None,
                    task: str = "multilabel") -> nn.Module:
    """Factory: pick the loss from ``training.loss`` for the given task.

    Multi-label / binary: 'bce' (default, optionally with ``pos_weight``) | 'focal'.
    Multi-class: 'ce' (default).
    """
    return get_task(task).build_criterion(
        train_cfg.get("loss"),
        pos_weight=pos_weight,
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        focal_alpha=train_cfg.get("focal_alpha", -1.0),
    )
