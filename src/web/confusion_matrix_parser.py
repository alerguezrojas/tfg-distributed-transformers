"""Parse confusion_matrix_TIMESTAMP.csv into a pivoted DataFrame per epoch."""

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
