"""Parser para batch_metrics_*.csv generado por BatchMonitorDecorator.

Formato actual (v3):
    epoch, batch, n_batches, running_loss, batch_loss, lr, batch_f1, batch_acc, batch_prec

Formato v2:
    epoch, batch, n_batches, running_loss, batch_loss, lr

Formato legacy (v1):
    epoch, batch, n_batches, running_loss

Todos los formatos son compatibles — las columnas ausentes se rellenan con NaN.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ALL_COLS = ["epoch", "batch", "n_batches", "running_loss", "batch_loss", "lr",
             "batch_f1", "batch_acc", "batch_prec", "global_batch"]


def parse_batch_csv(csv_path: Path) -> pd.DataFrame:
    """Lee el CSV de métricas por batch y devuelve un DataFrame normalizado.

    Columnas garantizadas:
        epoch, batch, n_batches, running_loss, batch_loss, lr,
        batch_f1, batch_acc, batch_prec, global_batch

    Las columnas no presentes en el CSV se rellenan con NaN (compat. legacy).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=_ALL_COLS)

    if df.empty:
        return pd.DataFrame(columns=_ALL_COLS)

    # Añadir columnas nuevas con NaN si faltan (compatibilidad con CSVs legacy/v2)
    for col in ("batch_loss", "lr", "batch_f1", "batch_acc", "batch_prec"):
        if col not in df.columns:
            df[col] = float("nan")

    # Índice global de batch (útil para mostrar toda la historia de training)
    if "global_batch" not in df.columns:
        df["global_batch"] = (df["epoch"] - 1) * df["n_batches"] + df["batch"]

    return df
