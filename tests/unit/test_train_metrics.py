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
