"""Compare — _common."""
from __future__ import annotations


import pandas as pd

from src.web.run_registry import RunInfo


def _prec(r: RunInfo) -> str:
    """Runs without a precision marker predate the selector → fp32."""
    return r.precision or "fp32"


def _has(d: pd.DataFrame, c: str) -> bool:
    return c in d.columns and d[c].notna().any()


