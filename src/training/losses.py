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
the metadata-based ``pos_weight`` computation lives in ``pos_weight_from_metadata``
(kept separate so the factory stays unit-testable without any file I/O).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def pos_weight_from_metadata(metadata_path: str, split: str = "train",
                             clamp_max: float = 100.0) -> torch.Tensor:
    """Compute BCE ``pos_weight`` from the training-split class frequencies.

    Reads ``metadata.parquet`` and counts, per CORINE class (in dataset order),
    how many patches carry it. Returns a tensor of shape (num_classes,).
    """
    import pandas as pd
    from src.data.dataset import CLASSES, CLASS_TO_IDX, SPLIT_MAP

    df = pd.read_parquet(metadata_path)
    df = df[df["split"] == SPLIT_MAP.get(split, split)]
    pos = torch.zeros(len(CLASSES))
    for labels in df["labels"]:
        for lab in labels:
            idx = CLASS_TO_IDX.get(lab)
            if idx is not None:
                pos[idx] += 1
    return pos_weight_from_counts(pos, len(df), clamp_max=clamp_max)


def build_criterion(train_cfg: dict, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Factory: pick the loss from ``training.loss`` ('bce' | 'focal').

    'bce'   → BCEWithLogitsLoss (optionally with ``pos_weight``)
    'focal' → FocalLoss(gamma=training.focal_gamma, alpha=training.focal_alpha)
    """
    loss_kind = str(train_cfg.get("loss", "bce")).lower()
    if loss_kind == "focal":
        return FocalLoss(
            gamma=train_cfg.get("focal_gamma", 2.0),
            alpha=train_cfg.get("focal_alpha", -1.0),
        )
    if loss_kind in ("bce", "bcewithlogits", "bce_with_logits"):
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(f"training.loss must be 'bce' or 'focal', got {loss_kind!r}")
