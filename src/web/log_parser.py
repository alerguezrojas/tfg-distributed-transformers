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
_SIMPLE_THRESHOLD = re.compile(r"threshold óptimo=([0-9.]+),\s*F1=([0-9.]+)")
# \w* permite cualquier subclase: Trainer, DDPTrainer, HeterogeneousDDPTrainer
_ENERGY_TRAIN = re.compile(r"\[energy\]\s+\w*Trainer\.train_epoch:\s+([0-9.]+)\s+J\s+\([0-9.]+ Wh\)\s+potencia media\s+([0-9.]+) W")
_ENERGY_EVAL = re.compile(r"\[energy\]\s+\w*Trainer\.eval_epoch:\s+([0-9.]+)\s+J\s+\(([0-9.]+) Wh\)\s+potencia media\s+([0-9.]+) W")
_TIMED_TRAIN = re.compile(r"\[timed\]\s+\w*Trainer\.train_epoch:\s+([0-9.]+)s")
_TIMED_EVAL = re.compile(r"\[timed\]\s+\w*Trainer\.eval_epoch:\s+([0-9.]+)s")

# ── Deep-trace patterns ───────────────────────────────────────────────────────
# Hay (al menos) dos variantes en el orden de los campos de la línea RESUMEN:
#   A) ... val_f1=X  best=X  val_acc=X | ...   (cluster, may 2026)
#   B) ... val_f1=X  val_acc=X  best=X | ...   (local, may 2026)
# Por eso NO se usa un único regex posicional: se ancla la línea RESUMEN y se
# extrae cada campo por nombre, independientemente del orden.
_DEEP_ANCHOR = re.compile(r"\[E(\d+)/\d+\]\s+══\s+RESUMEN")

# ── Legacy format (pre-refactor simple trace) ─────────────────────────────────
_LEGACY_LINE = re.compile(
    r"\[Epoch (\d+)/\d+\]\s+"
    r"train_loss=([0-9.]+)\s+train_f1=([0-9.]+)\s+train_acc=([0-9.]+)\s*\|\s*"
    r"val_loss=([0-9.]+)\s+val_f1=([0-9.]+)\s+best=[0-9.]+\s+val_acc=([0-9.]+)\s*\|\s*"
    r"time=([0-9.]+)s"
)

_COLS = [
    "epoch", "train_loss", "val_loss", "train_f1", "val_f1",
    "train_acc", "val_acc", "val_prec", "val_rec", "epoch_time",
    "optimal_threshold", "f1_at_threshold",
    "energy_train_j", "energy_eval_j", "energy_eval_wh",
    "power_train_w", "power_eval_w",
    "time_train_s", "time_eval_s",
]


def parse_log(log_path: Path) -> pd.DataFrame:
    """Return one-row-per-epoch DataFrame from a training log file."""
    text = log_path.read_text(errors="replace")
    is_deep = "train_deep" in log_path.name
    if is_deep:
        df = _parse_deep(text)
    elif _LEGACY_LINE.search(text):
        df = _parse_legacy(text)
    else:
        df = _parse_simple(text)
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
            continue

        m = _SIMPLE_THRESHOLD.search(line)
        if m:
            cur["optimal_threshold"] = float(m.group(1))
            cur["f1_at_threshold"] = float(m.group(2))
            continue

        m = _ENERGY_TRAIN.search(line)
        if m:
            cur["energy_train_j"] = float(m.group(1))
            cur["power_train_w"] = float(m.group(2))
            continue

        m = _ENERGY_EVAL.search(line)
        if m:
            cur["energy_eval_j"] = float(m.group(1))
            cur["energy_eval_wh"] = float(m.group(2))
            cur["power_eval_w"] = float(m.group(3))
            continue

        m = _TIMED_TRAIN.search(line)
        if m:
            cur["time_train_s"] = float(m.group(1))
            continue

        m = _TIMED_EVAL.search(line)
        if m:
            cur["time_eval_s"] = float(m.group(1))

    if cur:
        rows.append(cur)

    return _to_df(rows)


def _deep_field(name: str, line: str) -> float | None:
    """Extrae `name=<float>` de una línea, o None si no aparece."""
    m = re.search(rf"\b{name}=([0-9.]+)", line)
    return float(m.group(1)) if m else None


# Campo del log → columna del DataFrame
_DEEP_FIELDS = [
    ("train_loss", "train_loss"), ("train_f1", "train_f1"), ("train_acc", "train_acc"),
    ("val_loss", "val_loss"), ("val_f1", "val_f1"), ("val_acc", "val_acc"),
    ("val_prec", "val_prec"), ("val_rec", "val_rec"), ("time", "epoch_time"),
]


def _parse_deep(text: str) -> pd.DataFrame:
    rows: list[dict] = []
    for line in text.splitlines():
        a = _DEEP_ANCHOR.search(line)
        if not a:
            continue
        row: dict = {"epoch": int(a.group(1))}
        for field, col in _DEEP_FIELDS:
            val = _deep_field(field, line)
            if val is not None:
                row[col] = val
        rows.append(row)
    return _to_df(rows)


def _parse_legacy(text: str) -> pd.DataFrame:
    rows: list[dict] = []
    for line in text.splitlines():
        m = _LEGACY_LINE.search(line)
        if m:
            rows.append({
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "train_f1": float(m.group(3)),
                "train_acc": float(m.group(4)),
                "val_loss": float(m.group(5)),
                "val_f1": float(m.group(6)),
                "val_acc": float(m.group(7)),
                "epoch_time": float(m.group(8)),
            })
    return _to_df(rows)


def _to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in _COLS:
        if col not in df.columns:
            df[col] = float("nan") if col != "epoch" else None
    return df
