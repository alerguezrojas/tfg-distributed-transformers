"""Tests for the train-metric fix: train F1 must be computed against the ORIGINAL
0/1 targets, never the mixup-blended/smoothed labels used for the loss.

Regression: the old code thresholded the soft mixed labels at 0.5, fabricating
wrong "true" labels and systematically biasing the reported train F1 (which the
project's train-val overfitting narrative depends on).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer


class _ConstLogits(nn.Module):
    """Always predicts every class positive (sigmoid(+10) > 0.5).

    Uses a trainable bias so the loss has a grad_fn (backward works); the input
    enters the graph multiplied by zero so the output stays a constant +10.
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.full((19,), 10.0))

    def forward(self, x):
        zero = x.flatten(1).sum(dim=1, keepdim=True) * 0.0   # (N, 1) zeros, in graph
        return self.bias.unsqueeze(0) + zero                 # (N, 19) ≈ +10


def _ones_loader(n_samples=16, batch_size=4):
    x = torch.randn(n_samples, 3, 8, 8)
    y = torch.ones(n_samples, 19)            # every sample: all classes positive
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _make_trainer(tmp_path, **kw):
    model = _ConstLogits()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # lr=0 → predictions stay +10
    return Trainer(model, opt, None, torch.device("cpu"), checkpoint_dir=str(tmp_path), **kw)


def test_train_f1_uses_original_labels_not_mixed(tmp_path, monkeypatch):
    """With mixup corrupting labels, the reported train F1 must still reflect the
    original (all-ones) targets — so a model predicting all-ones scores F1=1.0."""
    import src.training.trainer as trainer_mod

    # mixup_batch corrupts the labels to all-zeros (and leaves images unchanged)
    monkeypatch.setattr(
        trainer_mod, "mixup_batch",
        lambda img, lab, a: (img, torch.zeros_like(lab)),
    )
    # alternate: batch 1 mixed, batch 2 clean, ... → some clean batches exist
    seq = iter([0.0, 1.0, 0.0, 1.0])
    monkeypatch.setattr(trainer_mod.random, "random", lambda: next(seq))

    trainer = _make_trainer(tmp_path, mixup_alpha=0.5)
    out = trainer.train_epoch(_ones_loader(n_samples=16, batch_size=4))
    # If the metric were computed on the mixed (zeroed) labels it would be ~0.67.
    assert out["f1"] == 1.0


def test_train_f1_unchanged_without_mixup(tmp_path):
    trainer = _make_trainer(tmp_path, mixup_alpha=0.0)
    out = trainer.train_epoch(_ones_loader())
    assert out["f1"] == 1.0          # all-ones preds vs all-ones labels


def test_train_epoch_survives_all_batches_mixed(tmp_path, monkeypatch):
    """Degenerate case (every batch mixed) must not crash on cat([])."""
    import src.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod.random, "random", lambda: 0.0)  # always mixed
    trainer = _make_trainer(tmp_path, mixup_alpha=0.5)
    out = trainer.train_epoch(_ones_loader())
    assert "f1" in out and 0.0 <= out["f1"] <= 1.0


def test_label_smoothing_does_not_bias_train_f1(tmp_path):
    """Label smoothing alone (no mixup) must leave the metric on true labels."""
    trainer = _make_trainer(tmp_path, label_smoothing=0.1)
    out = trainer.train_epoch(_ones_loader())
    assert out["f1"] == 1.0


def test_bare_trainer_fit_raises(tmp_path):
    """The training loop lives only in EpochController; a bare Trainer.fit must refuse."""
    import pytest
    trainer = _make_trainer(tmp_path)
    with pytest.raises(NotImplementedError):
        trainer.fit(_ones_loader(), _ones_loader(), epochs=1)


# ── Model selection: f1@0.5 vs f1@optimal-threshold (fairness for focal) ────────

from src.training.base_trainer import BaseTrainer
from src.training.decorators.base import EpochController


class _ScriptedTrainer(BaseTrainer):
    """Returns a scripted sequence of val metrics; records which epochs are saved."""

    def __init__(self, val_seq):
        self.val_seq = val_seq
        self.saved: list[int] = []
        self._i = 0

    def train_epoch(self, loader):
        return {"f1": 0.0, "loss": 0.0, "accuracy": 0.0, "time": 0.0}

    def eval_epoch(self, loader):
        m = self.val_seq[self._i]
        self._i += 1
        return m

    def save_checkpoint(self, epoch, metrics):
        self.saved.append(epoch)

    def fit(self, *a):
        pass


# Epoch 2 is best by optimal-threshold F1 (0.70) but NOT by 0.5-F1 (0.40 < 0.50).
_SEQ = [
    {"f1": 0.50, "_f1_at_optimal_threshold": 0.50},
    {"f1": 0.40, "_f1_at_optimal_threshold": 0.70},
    {"f1": 0.45, "_f1_at_optimal_threshold": 0.60},
]


def test_selection_by_f1_threshold_05():
    t = _ScriptedTrainer([dict(d) for d in _SEQ])
    EpochController(t, patience=None, select_metric="f1").fit(None, None, epochs=3)
    assert t.saved == [1]                 # only epoch 1 beats the 0.5-F1 baseline


def test_selection_by_optimal_threshold():
    t = _ScriptedTrainer([dict(d) for d in _SEQ])
    EpochController(t, patience=None, select_metric="f1_optimal").fit(None, None, epochs=3)
    assert t.saved == [1, 2]              # epoch 2 wins on optimal-threshold F1


def test_selection_default_is_f1():
    t = _ScriptedTrainer([dict(d) for d in _SEQ])
    EpochController(t, patience=None).fit(None, None, epochs=3)
    assert t.saved == [1]                 # back-compat: default selects by 0.5-F1
