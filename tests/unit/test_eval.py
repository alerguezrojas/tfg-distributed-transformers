"""Unit tests for scripts/eval.py — the held-out test-set evaluator.

The script was broken (it called build_model with an unsupported ``dropout``
kwarg) and had never been run. These tests lock in the pure helpers so the
test-set number that feeds the dashboard stays correct.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location("eval_script", ROOT / "scripts" / "eval.py")
eval_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_script)


def test_threshold_search_picks_best_macro_f1():
    # Both classes recoverable (each positive in one row) → macro F1 = 1 at a
    # threshold between the low and high probs.
    probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    labels = torch.tensor([[1, 0], [0, 1]])
    t, f1 = eval_script.threshold_search(probs, labels)
    assert f1 > 0.99
    assert t < 0.9


def test_per_class_metrics_shape_and_perfect_class():
    probs = torch.tensor([[0.9, 0.1], [0.7, 0.2], [0.95, 0.05]])
    labels = torch.tensor([[1, 0], [1, 0], [1, 0]])
    rows = eval_script.per_class_metrics(probs, labels, threshold=0.5)
    assert len(rows) == 2
    # class 0 is always present and always predicted → F1 = 1
    assert rows[0]["f1"] > 0.99
    # class 1 never present nor predicted → F1 = 0
    assert rows[1]["f1"] == 0.0
    assert set(rows[0]) == {"class_idx", "class_name", "f1", "precision", "recall"}


def test_evaluate_respects_max_batches():
    class _Tiny(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 2)

    # 5 batches of 4 available; cap at 2.
    data = [(torch.zeros(4, 3, 8, 8), torch.zeros(4, 2)) for _ in range(5)]
    probs, labels, loss = eval_script.evaluate(_Tiny(), data, torch.device("cpu"), max_batches=2)
    assert probs.shape[0] == 8           # 2 batches × 4
    assert labels.shape[0] == 8
    assert loss == loss                  # not NaN (divided by n_batches, not len)
