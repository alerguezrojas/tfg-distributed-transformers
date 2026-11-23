"""Aggregate held-out test results over repeated runs (mean ± sample std).

Each run folder ``logs/{env}/[dataset/]{mode}/{model}/`` holds, per run, a
``train_<ts>.log`` and the ``test_<ts>.csv`` written by ``scripts/eval.py``. A run
counts when its log records the seed in the ``Configuración:`` line, or else its
manifest ``run_<ts>.json`` does (so earlier runs without a seed are left out) and its test
CSV exists.

Pure (no torch): reused by ``scripts/summarize_runs.py`` and testable on tmp files.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from src.web.eval_parser import parse_eval_csv

METRICS = ("auc", "accuracy", "f1_t05", "f1_opt", "qwk")   # qwk: ordinal datasets (PANDA)
_TS = re.compile(r"train_(\d{8}_\d{6})\.log$")


@dataclass
class RunResult:
    folder: Path
    timestamp: str
    seed: int
    metrics: dict[str, float]


def _config_line(log: Path) -> dict[str, str]:
    for line in log.read_text(errors="replace").splitlines():
        i = line.find("Configuración:")
        if i >= 0:
            return {k.strip(): v.strip() for k, v in
                    (p.split("=", 1) for p in line[i + 14:].split("|") if "=" in p)}
    return {}


def _manifest_seed(folder: Path, ts: str) -> str:
    """Seed from the run manifest ``run_<ts>.json`` (``config.training.seed``), the fallback
    for logs whose ``Configuración:`` line carries none; ``"None"`` if absent."""
    try:
        data = json.loads((folder / f"run_{ts}.json").read_text(encoding="utf-8"))
        seed = data["config"]["training"]["seed"]
    except (OSError, ValueError, KeyError, TypeError):
        return "None"
    return "None" if seed is None else str(seed)


def collect(folder: Path) -> list[RunResult]:
    """Seeded runs of one folder that have a test CSV."""
    runs = []
    for log in sorted(folder.glob("train_*.log")):
        m = _TS.search(log.name)
        seed = _config_line(log).get("seed", "None")
        if seed == "None" and m:
            seed = _manifest_seed(folder, m.group(1))
        test_csv = folder / f"test_{m.group(1)}.csv" if m else None
        if not m or seed == "None" or not test_csv.exists():
            continue
        _, agg = parse_eval_csv(test_csv)
        runs.append(RunResult(folder, m.group(1), int(seed),
                              {k: float(agg[k]) for k in METRICS if k in agg}))
    return runs


def summarize(runs: list[RunResult]) -> dict[str, tuple[float, float, int]]:
    """metric → (mean, sample std, n) over the runs that report it."""
    out = {}
    for k in METRICS:
        values = [r.metrics[k] for r in runs if k in r.metrics]
        if values:
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            out[k] = (statistics.fmean(values), std, len(values))
    return out
