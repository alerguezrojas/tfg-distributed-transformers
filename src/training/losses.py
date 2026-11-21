"""Loss functions for multi-label classification — levers against the macro-F1
ceiling caused by class imbalance (rare CORINE classes the model never predicts).

Two interchangeable options, both compatible with soft targets (mixup) since they
operate element-wise on targets in [0, 1]:

  • ``BCEWithLogitsLoss(pos_weight=…)`` — re-weights the positive term of each
    class by neg/pos, so rare positives count more. ``pos_weight='auto'`` in the
    config computes the weights from the training-split class frequencies.
  • ``FocalLoss`` (Lin et al. 2017, multi-label variant) — down-weights easy
    examples via ``(1-p_t)^gamma`` so the loss focuses on the hard, rare ones.

For multi-class tasks with ordered classes (a cancer grade) there is also
``CrossEntropyEMDLoss`` (``loss: ce_emd``): cross-entropy plus the squared Earth
Mover's Distance between the predicted and the target class distributions (Hou et
al. 2017), so predicting a grade several steps away costs more than a neighbour.

Against noisy labels, any of these losses can be wrapped in ``OUSMLoss``
(``training.ousm_k``): each training batch ignores its k highest-loss samples.

``build_criterion`` is a pure factory selected by ``training.loss`` in the config;
which losses are valid depends on the task (see :mod:`src.training.tasks`). The
class frequencies behind ``pos_weight: auto`` come from the dataset spec
(``DatasetSpec.class_counts``), so this module stays free of any file I/O.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.tasks import Task, get_task


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


def squared_emd(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-sample squared Earth Mover's Distance between ``softmax(logits)`` and
    ``targets`` (N, C) over ORDERED classes: ``sum_k (CDF_pred(k) - CDF_target(k))^2``.

    The targets may be soft (one-hot, mixup blends, label smoothing). Moving
    probability mass by d classes costs ~d, so far-off grades are penalized more
    than neighbours; the value is in [0, C-1]. Computed in float32 (stable under AMP).
    """
    probs = torch.softmax(logits.float(), dim=1)
    diff = torch.cumsum(probs, dim=1) - torch.cumsum(targets.float(), dim=1)
    return (diff ** 2).sum(dim=1)


class CrossEntropyEMDLoss(nn.Module):
    """Cross-entropy + ``emd_weight`` x squared EMD, for ordered classes.

    The cross-entropy keeps the usual per-class training signal and the EMD term
    (Hou, Yu & Samaras, 2017, arXiv:1611.05916) adds the order of the classes, the
    same idea as the quadratic weighted kappa used to evaluate graded problems.
    Same head and predictions as plain multi-class (argmax).
    """

    def __init__(self, emd_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if not math.isfinite(emd_weight) or emd_weight < 0:
            raise ValueError(f"emd_weight must be a finite number >= 0, got {emd_weight}")
        self.emd_weight = float(emd_weight)
        self.reduction = reduction            # 'mean' | 'none' (per sample, for OUSMLoss)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.reduction == "none":
            return (F.cross_entropy(logits, targets, reduction="none")
                    + self.emd_weight * squared_emd(logits, targets))
        return (F.cross_entropy(logits, targets)
                + self.emd_weight * squared_emd(logits, targets).mean())


class OUSMLoss(nn.Module):
    """Online Uncertainty Sample Mining: the batch loss without its ``k`` worst samples.

    Xue, Dou, Shi, Chen & Heng (2019), *Robust learning at noisy labeled medical images:
    applied to skin lesion classification* (ISBI, arXiv:1901.07759); used against the
    noisy ISUP grades of PANDA by the 6th-placed team (``docs/research/panda.md``). A
    sample whose label is wrong tends to keep a high loss, so leaving the ``k`` highest
    losses of each batch out of the mean keeps those labels from pulling the model.
    This is the plain top-k exclusion; the paper also re-weights the hard samples it keeps,
    which is not implemented here.

    ``base`` is any loss with a ``reduction`` attribute (CE, CE + EMD, BCE, focal); it is
    switched to per-element output and averaged per sample. Mining only happens while
    gradients are being recorded (the training step): under ``torch.no_grad()`` (the
    validation loop) the plain mean of every sample is returned, so the validation loss
    stays comparable with runs without OUSM. At least one sample is always kept. Under
    DDP each rank drops ``k`` from its own share of the batch.
    """

    def __init__(self, base: nn.Module, k: int):
        super().__init__()
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError(f"ousm_k must be an integer >= 0, got {k!r}")
        if not hasattr(base, "reduction"):
            raise ValueError(f"{type(base).__name__} has no 'reduction': OUSM needs "
                             f"per-sample losses")
        base.reduction = "none"
        self.base = base
        self.k = k

    @property
    def pos_weight(self) -> torch.Tensor | None:
        """The wrapped loss's ``pos_weight`` (read by the heterogeneous trainer)."""
        return getattr(self.base, "pos_weight", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_sample = self.base(logits, targets)
        if per_sample.dim() > 1:                   # (N, C) element losses -> (N,)
            per_sample = per_sample.flatten(1).mean(dim=1)
        drop = min(self.k, per_sample.numel() - 1) if torch.is_grad_enabled() else 0
        if drop <= 0:
            return per_sample.mean()
        kept = torch.topk(per_sample, per_sample.numel() - drop, largest=False).values
        return kept.mean()


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
                    task: str | Task = "multilabel") -> nn.Module:
    """Factory: pick the loss from ``training.loss`` for the given task.

    Multi-label / binary: 'bce' (default, optionally with ``pos_weight``) | 'focal'.
    Multi-class: 'ce' (default); with ordered classes also 'ce_emd'
    (``training.emd_weight``, default 1.0). ``training.ousm_k`` > 0 wraps the loss in
    :class:`OUSMLoss` (the k highest-loss samples of each training batch are ignored).
    ``task`` is a task name or a
    :class:`Task` instance — pass the dataset's own task (``spec.make_task()``) so
    that its ordered classes are known.
    """
    task_obj = task if isinstance(task, Task) else get_task(task)
    criterion = task_obj.build_criterion(
        train_cfg.get("loss"),
        pos_weight=pos_weight,
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        focal_alpha=train_cfg.get("focal_alpha", -1.0),
        emd_weight=train_cfg.get("emd_weight", 1.0),
    )
    ousm_k = train_cfg.get("ousm_k") or 0
    return OUSMLoss(criterion, ousm_k) if ousm_k else criterion


def configured_loss_name(train_cfg: dict, task) -> str:
    """Name of the loss in effect: ``training.loss`` or, if unset, the task's default.

    Used for the ``loss=`` label of the log's ``Configuración:`` line, so it matches
    what :func:`build_criterion` really builds ('ce' for multiclass, 'bce' otherwise).
    """
    return str(train_cfg.get("loss") or task.default_loss).lower()
