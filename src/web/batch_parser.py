"""Parser para batch_metrics_*.csv generado por BatchMonitorDecorator.

Formato actual (v2):
    epoch, batch, n_batches, running_loss, batch_loss, lr

Formato legacy (v1):
    epoch, batch, n_batches, running_loss

Ambos formatos son compatibles — las columnas nuevas tienen NaN en CSVs legacy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_batch_csv(csv_path: Path) -> pd.DataFrame:
    """Lee el CSV de métricas por batch y devuelve un DataFrame normalizado.

    Columnas garantizadas:
        epoch, batch, n_batches, running_loss, batch_loss, lr, global_batch

    batch_loss y lr tendrán NaN para CSVs legacy que no los incluyen.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=["epoch", "batch", "n_batches",
                                     "running_loss", "batch_loss", "lr", "global_batch"])

    if df.empty:
        return pd.DataFrame(columns=["epoch", "batch", "n_batches",
                                     "running_loss", "batch_loss", "lr", "global_batch"])

    # Añadir columnas nuevas con NaN si faltan (compatibilidad con CSVs legacy)
    for col in ("batch_loss", "lr"):
        if col not in df.columns:
            df[col] = float("nan")

    # Índice global de batch (útil para mostrar toda la historia de training)
    if "global_batch" not in df.columns:
        df["global_batch"] = (df["epoch"] - 1) * df["n_batches"] + df["batch"]

    return df
