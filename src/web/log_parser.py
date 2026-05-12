"""Parse training log files into pandas DataFrames."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ── Simple-trace patterns ────────────────────────────────────────────────────
_SIMPLE_EPOCH = re.compile(r"── Epoch (\d+)/(\d+)")
_SIMPLE_METRIC = re.compile(r"\s+(loss|f1|accuracy)\s+train=([0-9.]+)\s+val=([0-9.]+)")
_SIMPLE_PRECISION = re.compile(r"\s+precision\s+val=([0-9.]+)")
_SIMPLE_RECALL = re.compile(r"\s+recall\s+val=([0-9.]+)")
_SIMPLE_ETA = re.compile(r"ETA:.*?\(([0-9.]+)s/epoch")

# ── Deep-trace patterns ───────────────────────────────────────────────────────
_DEEP_RESUMEN = re.compile(
    r"\[E(\d+)/\d+\] ══ RESUMEN\s+"
    r"train_loss=([0-9.]+)\s+train_f1=([0-9.]+)\s+train_acc=([0-9.]+)\s*\|\s*"
    r"val_loss=([0-9.]+)\s+val_f1=([0-9.]+)\s+best=[0-9.]+\s*\|\s*"
    r"val_prec=([0-9.]+)\s+val_rec=([0-9.]+)\s*\|\s*"
    r"time=([0-9.]+)s"
)

_COLS = ["epoch", "train_loss", "val_loss", "train_f1", "val_f1",
         "train_acc", "val_acc", "val_prec", "val_rec", "epoch_time"]


def parse_log(log_path: Path) -> pd.DataFrame:
    """Return one-row-per-epoch DataFrame from a training log file."""
    text = log_path.read_text(errors="replace")
    is_deep = "train_deep" in log_path.name
    df = _parse_deep(text) if is_deep else _parse_simple(text)
    return df[_COLS]


def _parse_simple(text: str) -> pd.DataFrame:
    rows: list[dict] = []
    cur: dict = {}

    for line in text.splitlines():
        m = _SIMPLE_EPOCH.search(line)
        if m:
            if cur:
                rows.append(cur)
            cur = {"epoch": int(m.group(1))}
            continue

        if not cur:
            continue

        m = _SIMPLE_METRIC.search(line)
        if m:
            raw_key = m.group(1)
            key = "acc" if raw_key == "accuracy" else raw_key
            cur[f"train_{key}"] = float(m.group(2))
            cur[f"val_{key}"] = float(m.group(3))
            continue

        m = _SIMPLE_PRECISION.search(line)
        if m:
            cur["val_prec"] = float(m.group(1))
            continue

        m = _SIMPLE_RECALL.search(line)
        if m:
            cur["val_rec"] = float(m.group(1))
            continue

        m = _SIMPLE_ETA.search(line)
        if m:
            cur["epoch_time"] = float(m.group(1))

    if cur:
        rows.append(cur)

    return _to_df(rows)


def _parse_deep(text: str) -> pd.DataFrame:
    rows: list[dict] = []
    for line in text.splitlines():
        m = _DEEP_RESUMEN.search(line)
        if m:
            rows.append({
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "train_f1": float(m.group(3)),
                "train_acc": float(m.group(4)),
                "val_loss": float(m.group(5)),
                "val_f1": float(m.group(6)),
                "val_prec": float(m.group(7)),
                "val_rec": float(m.group(8)),
                "epoch_time": float(m.group(9)),
            })
    return _to_df(rows)


def _to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in _COLS:
        if col not in df.columns:
            df[col] = float("nan") if col != "epoch" else None
    return df
