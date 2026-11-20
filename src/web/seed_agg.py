"""Aggregation of runs that differ only in the seed: mean ± std and per-epoch bands.

A seed changes the initialization and the batch order, so one run is one draw; the
honest number of a configuration is the mean over seeds with its spread (roadmap:
"Agregación por semilla"). Pure (pandas/NumPy): no Streamlit, so it is unit-tested.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import pandas as pd


def group_runs(runs: Sequence) -> list[list]:
    """Runs grouped by ``RunInfo.seed_group_key`` (order of first appearance kept)."""
    groups: dict[tuple, list] = {}
    for r in runs:
        groups.setdefault(r.seed_group_key, []).append(r)
    return list(groups.values())


def mean_std(values: Iterable[float]) -> tuple[float, float, int]:
    """(mean, sample std, n) of the finite values; std is 0 for a single value and the
    mean is NaN for none."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan"), 0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def fmt_mean_std(values: Iterable[float], digits: int = 4) -> str:
    """'0.7216 ± 0.0112', just the value for a single run, '—' for none."""
    mean, std, n = mean_std(values)
    if n == 0:
        return "—"
    return f"{mean:.{digits}f}" if n == 1 else f"{mean:.{digits}f} ± {std:.{digits}f}"


def run_numbers(df: pd.DataFrame, column: str) -> dict[str, float]:
    """Headline numbers of one run's epoch table for the metric ``column``:
    ``best`` (max) and its ``best_epoch``, the ``final`` value, ``epochs`` run and the
    training ``duration_s``. Missing data is left out."""
    out: dict[str, float] = {}
    if df is None or df.empty:
        return out
    out["epochs"] = float(len(df))
    if column in df.columns and "epoch" in df.columns:
        valid = df[["epoch", column]].dropna()
        if not valid.empty:
            top = valid.loc[valid[column].idxmax()]
            out["best"], out["best_epoch"] = float(top[column]), float(top["epoch"])
            out["final"] = float(valid.sort_values("epoch")[column].iloc[-1])
    if "epoch_time" in df.columns and df["epoch_time"].notna().any():
        out["duration_s"] = float(df["epoch_time"].dropna().sum())
    return out


def curve_band(dfs: Sequence[pd.DataFrame], column: str) -> pd.DataFrame:
    """Per-epoch mean and std of ``column`` over runs → columns epoch, mean, std, n, lo, hi.

    Runs stop at different epochs (early stopping): each epoch averages the runs that
    reached it, and ``n`` says how many. ``lo``/``hi`` = mean ∓ std.
    """
    parts = [d[["epoch", column]].dropna().set_index("epoch")[column]
             for d in dfs if column in d.columns and "epoch" in d.columns]
    if not parts:
        return pd.DataFrame(columns=["epoch", "mean", "std", "n", "lo", "hi"])
    wide = pd.concat(parts, axis=1).sort_index()
    out = pd.DataFrame({
        "mean": wide.mean(axis=1),
        "std": wide.std(axis=1, ddof=1).fillna(0.0),     # one run → no spread
        "n": wide.notna().sum(axis=1),
    })
    out["lo"], out["hi"] = out["mean"] - out["std"], out["mean"] + out["std"]
    out.index.name = "epoch"
    return out.reset_index()


@dataclass
class Series:
    """What the Compare page draws as one entry: a single run, or the runs of one
    configuration that differ only in the seed (mean ± std over them)."""

    label: str
    runs: list
    dfs: list[pd.DataFrame]

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def seeds(self) -> list[int]:
        return sorted(r.meta.seed for r in self.runs if r.meta and r.meta.seed is not None)


def build_series(runs: Sequence, df_by_label: dict[str, pd.DataFrame],
                 aggregate: bool = True) -> list[Series]:
    """One :class:`Series` per run, or per seed group when ``aggregate`` (a group of a
    single run keeps the run's own label)."""
    groups = group_runs(runs) if aggregate else [[r] for r in runs]
    out = []
    for g in groups:
        label = g[0].label if len(g) == 1 else f"{g[0].group_label} ×{len(g)}"
        out.append(Series(label, g, [df_by_label[r.label] for r in g]))
    return out
