"""Learning tasks — the kind of prediction problem a dataset poses.

A :class:`Task` (Strategy pattern) owns everything that depends on the problem
type and NOT on the dataset or on the distribution strategy:

  • the loss, and how label smoothing softens the training targets,
  • how logits become probabilities and hard 0/1 predictions,
  • the metric definitions (accuracy differs between multi-label and multi-class),
  • whether a decision threshold can be tuned on validation.

Targets are always float tensors of shape (N, C): multi-hot for multi-label,
one-hot for multi-class and (N, 1) for binary. A single shape keeps mixup, the
trainers and the decorator stack task-agnostic — they ask the task instead of
assuming sigmoid/0.5.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training import metrics as m

_DEFAULT_THRESHOLD = 0.5


class Task(ABC):
    """A prediction problem type. Stateless: instances are freely shareable."""

    name: ClassVar[str]
    #: Element-wise sigmoid outputs (True) vs. a softmax over the classes (False).
    #: Losses that sum per (sample, class) element — the heterogeneous DDP gradient
    #: normalization — are only defined for sigmoid tasks.
    sigmoid_outputs: ClassVar[bool]
    #: Whether an optimal decision threshold can be searched on validation.
    searches_threshold: ClassVar[bool]
    default_loss: ClassVar[str]

    # ── Loss ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def build_criterion(self, loss: str | None = None, *,
                        pos_weight: torch.Tensor | None = None,
                        focal_gamma: float = 2.0,
                        focal_alpha: float = -1.0) -> nn.Module:
        """Loss module for ``training.loss`` (None → the task default)."""

    @abstractmethod
    def smooth_targets(self, targets: torch.Tensor, eps: float) -> torch.Tensor:
        """Label-smoothed copy of the (possibly mixup-blended) targets."""

    # ── Predictions ───────────────────────────────────────────────────────────

    @abstractmethod
    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits (N, C) → probabilities (N, C)."""

    @abstractmethod
    def predict(self, probs: torch.Tensor,
                threshold: float = _DEFAULT_THRESHOLD) -> torch.Tensor:
        """Probabilities → hard 0/1 predictions (long, N × C)."""

    @abstractmethod
    def hard_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Targets (possibly soft) → hard 0/1 ground truth (long, N × C)."""

    # ── Metrics ───────────────────────────────────────────────────────────────

    @abstractmethod
    def accuracy(self, preds: torch.Tensor, hard: torch.Tensor) -> float:
        """Task-appropriate accuracy on hard predictions and targets."""

    def metrics(self, preds: torch.Tensor, hard: torch.Tensor) -> dict[str, float]:
        """Macro F1/precision/recall (per-class columns) + task accuracy."""
        return {
            "f1": m.f1_score(preds, hard),
            "accuracy": self.accuracy(preds, hard),
            "precision": m.precision(preds, hard),
            "recall": m.recall(preds, hard),
        }

    def random_targets(self, n: int, num_classes: int) -> torch.Tensor:
        """Synthetic float targets (n, C) in this task's format (for benchmarks)."""
        return torch.randint(0, 2, (n, num_classes)).float()

    def search_threshold(self, probs: torch.Tensor,
                         hard: torch.Tensor) -> tuple[float, float] | None:
        """(best threshold, macro F1 at it) over ``metrics.THRESHOLD_GRID``, or
        None when the task has no tunable threshold."""
        return None


class MultiLabelTask(Task):
    """Several independent yes/no labels per sample (e.g. BigEarthNet, ChestX-ray14)."""

    name = "multilabel"
    sigmoid_outputs = True
    searches_threshold = True
    default_loss = "bce"

    _BCE_ALIASES = ("bce", "bcewithlogits", "bce_with_logits")

    def build_criterion(self, loss=None, *, pos_weight=None,
                        focal_gamma=2.0, focal_alpha=-1.0) -> nn.Module:
        from src.training.losses import FocalLoss
        kind = str(loss or self.default_loss).lower()
        if kind in self._BCE_ALIASES:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if kind == "focal":
            return FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        raise ValueError(f"loss {kind!r} is not valid for a {self.name} task "
                         f"(use 'bce' or 'focal')")

    def smooth_targets(self, targets, eps):
        # Each label is its own binary problem → smooth towards 0.5.
        return targets * (1.0 - eps) + 0.5 * eps

    def probabilities(self, logits):
        return torch.sigmoid(logits)

    def predict(self, probs, threshold=_DEFAULT_THRESHOLD):
        return (probs > threshold).long()

    def hard_targets(self, targets):
        return (targets > 0.5).long()

    def accuracy(self, preds, hard):
        return m.accuracy(preds, hard)       # sample-averaged label accuracy

    def search_threshold(self, probs, hard):
        best_t = _DEFAULT_THRESHOLD
        best_f1 = m.f1_score(self.predict(probs, best_t), hard)
        for t in m.THRESHOLD_GRID:
            f1_t = m.f1_score(self.predict(probs, t), hard)
            if f1_t > best_f1:
                best_t, best_f1 = t, f1_t
        return best_t, best_f1


class BinaryTask(MultiLabelTask):
    """One yes/no label per sample (e.g. tumour present) — a single sigmoid output.

    Mathematically the one-column case of multi-label, so it reuses its loss,
    predictions and threshold search; macro metrics reduce to the positive class.
    """

    name = "binary"


class MultiClassTask(Task):
    """Exactly one class per sample (e.g. lesion type) — softmax over the classes."""

    name = "multiclass"
    sigmoid_outputs = False
    searches_threshold = False
    default_loss = "ce"

    _CE_ALIASES = ("ce", "cross_entropy", "crossentropy")

    def build_criterion(self, loss=None, *, pos_weight=None,
                        focal_gamma=2.0, focal_alpha=-1.0) -> nn.Module:
        kind = str(loss or self.default_loss).lower()
        if kind not in self._CE_ALIASES:
            raise ValueError(f"loss {kind!r} is not valid for a {self.name} task "
                             f"(use 'ce')")
        if pos_weight is not None:
            raise ValueError("pos_weight applies to sigmoid (multi-label/binary) "
                             "tasks only")
        # Soft (one-hot / mixed / smoothed) probability targets are supported.
        return nn.CrossEntropyLoss()

    def smooth_targets(self, targets, eps):
        # Spread eps uniformly over the C classes (standard multi-class smoothing).
        return targets * (1.0 - eps) + eps / targets.shape[1]

    def probabilities(self, logits):
        return torch.softmax(logits, dim=1)

    def predict(self, probs, threshold=_DEFAULT_THRESHOLD):
        return F.one_hot(probs.argmax(dim=1), probs.shape[1]).long()

    def hard_targets(self, targets):
        return F.one_hot(targets.argmax(dim=1), targets.shape[1]).long()

    def accuracy(self, preds, hard):
        return (preds.argmax(dim=1) == hard.argmax(dim=1)).float().mean().item()

    def random_targets(self, n, num_classes):
        return F.one_hot(torch.randint(0, num_classes, (n,)), num_classes).float()


TASKS: dict[str, type[Task]] = {
    t.name: t for t in (MultiLabelTask, BinaryTask, MultiClassTask)
}


def get_task(name: str) -> Task:
    """Task instance by name ('multilabel' | 'binary' | 'multiclass')."""
    try:
        return TASKS[name]()
    except KeyError:
        raise ValueError(f"unknown task {name!r}; available: {sorted(TASKS)}") from None
