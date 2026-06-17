"""Unit tests for the multi-label confusion helpers."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.web.confusion_matrix_parser import (
    recall_by_class, top_confusions, confusion_profile,
)


def _df():
    # 3 classes, epoch 1. Diagonal = recall; off-diagonal = co-activation.
    rows = [
        (1, "A", "A", 0.90), (1, "A", "B", 0.40), (1, "A", "C", 0.05),
        (1, "B", "A", 0.30), (1, "B", "B", 0.50), (1, "B", "C", 0.02),
        (1, "C", "A", 0.10), (1, "C", "B", 0.20), (1, "C", "C", 0.10),
    ]
    return pd.DataFrame(rows, columns=["epoch", "true_class", "pred_class", "value"])


def test_recall_by_class_is_the_diagonal_sorted():
    rec = recall_by_class(_df(), 1)
    assert rec["A"] == 0.90 and rec["B"] == 0.50 and rec["C"] == 0.10
    # sorted ascending → worst class first
    assert list(rec.index) == ["C", "B", "A"]


def test_top_confusions_excludes_diagonal_and_sorts():
    top = top_confusions(_df(), 1, k=3)
    # diagonal never appears
    assert not ((top["true_class"] == top["pred_class"]).any())
    # strongest off-diagonal is A→B (0.40)
    assert top.iloc[0]["true_class"] == "A" and top.iloc[0]["pred_class"] == "B"
    assert top.iloc[0]["value"] == 0.40


def test_top_confusions_min_value_filters():
    top = top_confusions(_df(), 1, k=99, min_value=0.25)
    assert set(zip(top["true_class"], top["pred_class"])) == {("A", "B"), ("B", "A")}


def test_confusion_profile_excludes_self():
    prof = confusion_profile(_df(), 1, "A")
    assert "A" not in prof.index
    assert prof.index[0] == "B" and prof.iloc[0] == 0.40
