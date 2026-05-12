"""Read batch-level metrics CSV files produced by BatchMonitorDecorator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_batch_csv(csv_path: Path) -> pd.DataFrame:
    """Return DataFrame with columns: epoch, batch, n_batches, running_loss."""
    df = pd.read_csv(csv_path)
    df["global_batch"] = (df["epoch"] - 1) * df["n_batches"] + df["batch"]
    return df
