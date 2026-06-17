"""Parse and interpret confusion_matrix_TIMESTAMP.csv.

The task is MULTI-LABEL (each image may carry several of the 19 CORINE classes),
so a classic N×N confusion matrix does not apply. What the decorator stores is a
label co-activation matrix, normalized per true class:

    cell(i, j) = P(model predicts j | class i is truly present)

Reading it:
  * diagonal  cell(i, i)  = recall of class i (how often the true class is caught)
  * off-diag  cell(i, j)  = when i is present, how often label j ALSO fires —
                            a mix of genuine confusion and natural co-occurrence
                            (e.g. forest types co-occur; that is expected).

The helpers below turn that matrix into the digestible views the dashboard shows
(recall per class, strongest confusions, per-class confusion profile).
"""

from pathlib import Path

import pandas as pd


def parse_confusion_matrix_csv(csv_path: Path) -> pd.DataFrame:
    """Return DataFrame with columns: epoch, true_class, pred_class, value."""
    df = pd.read_csv(csv_path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def get_matrix_for_epoch(df: pd.DataFrame, epoch: int) -> pd.DataFrame:
    """Return a 19×19 pivot table (true_class × pred_class) for a given epoch."""
    ep_df = df[df["epoch"] == epoch]
    return ep_df.pivot(index="true_class", columns="pred_class", values="value")


def recall_by_class(df: pd.DataFrame, epoch: int) -> pd.Series:
    """Diagonal of the matrix = recall per class, indexed by class name (ascending)."""
    ep = df[(df["epoch"] == epoch) & (df["true_class"] == df["pred_class"])]
    return ep.set_index("true_class")["value"].sort_values()


def top_confusions(df: pd.DataFrame, epoch: int, k: int = 10,
                   min_value: float = 0.05) -> pd.DataFrame:
    """Strongest off-diagonal cells: 'when true_class is present, the model also
    predicts pred_class with this frequency'. Sorted descending."""
    ep = df[(df["epoch"] == epoch) & (df["true_class"] != df["pred_class"])].copy()
    ep = ep[ep["value"] >= min_value]
    ep = ep.sort_values("value", ascending=False).head(k)
    return ep[["true_class", "pred_class", "value"]].reset_index(drop=True)


def confusion_profile(df: pd.DataFrame, epoch: int, true_class: str) -> pd.Series:
    """For one true class, the labels the model ALSO fires (off-diagonal row),
    sorted descending. Answers 'when X is present, what else gets predicted?'"""
    ep = df[(df["epoch"] == epoch)
            & (df["true_class"] == true_class)
            & (df["pred_class"] != true_class)]
    return ep.set_index("pred_class")["value"].sort_values(ascending=False)
