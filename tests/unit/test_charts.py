"""Tests for the dashboard chart helpers: series naming in _metric_fig."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.web.ui.charts import _metric_fig

_DF = pd.DataFrame({
    "epoch": [1, 2, 3],
    "train_f1": [0.40, 0.50, 0.55],
    "val_f1": [0.35, 0.45, 0.48],
    "val_prec": [0.50, 0.56, 0.60],
    "val_rec": [0.30, 0.38, 0.45],
})


def test_metric_fig_default_names_are_train_val():
    fig = _metric_fig(_DF, "train_f1", "val_f1", "F1 (macro)", "F1")
    assert [t.name for t in fig.data] == ["Train", "Val"]


def test_metric_fig_custom_names_for_val_only_pairs():
    # Precision & Recall are both validation curves: the legend must not
    # label them Train/Val (regression guard for the mislabelled chart).
    fig = _metric_fig(
        _DF, "val_prec", "val_rec", "Precision & Recall (val)", "Score",
        name_train="Precision", name_val="Recall",
    )
    assert [t.name for t in fig.data] == ["Precision", "Recall"]


def test_metric_fig_skips_missing_columns():
    df = _DF.drop(columns=["train_f1"])
    fig = _metric_fig(df, "train_f1", "val_f1", "F1 (macro)", "F1")
    assert [t.name for t in fig.data] == ["Val"]
