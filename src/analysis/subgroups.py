"""Performance by subgroup (centre, sex, age band...) with the gap between groups.

A model can do well overall and fail in a clinically relevant subset (Oakden-Rayner et al. 2020,
"Hidden stratification causes clinically meaningful failures in machine learning for medical
imaging", ACM CHIL; arXiv 1909.12475); TRIPOD+AI (Collins et al. 2024, BMJ 385:e078378) asks for
performance estimates with confidence intervals for the key subgroups. This module computes the
standard metrics for every value of a group label, optionally with the bootstrap intervals of
``bootstrap.py``, and the gap between the best and the worst group.
"""
import numpy as np

from src.analysis._common import check_pair, task_name
from src.analysis.bootstrap import point_estimates, metrics_with_ci

#: Metrics computed per group (names as in ``bootstrap.METRIC_NAMES``).
SUBGROUP_METRICS = ("auc", "f1", "sensitivity", "specificity", "ece")


def _label(value):
    return value.item() if hasattr(value, "item") else value


def _check_groups(groups, n: int) -> np.ndarray:
    g = np.asarray(groups)
    if g.ndim != 1 or g.shape[0] != n:
        raise ValueError(f"groups must be 1-D with one label per sample ({n}); got {g.shape}")
    if g.dtype.kind == "f" and np.isnan(g).any():
        raise ValueError("groups contains NaN: encode the missing values as their own label")
    return g


def _entry(label, p, y, name, threshold, bins, n_boot, alpha, seed, stratified, small) -> dict:
    est = point_estimates(p, y, name, threshold, bins, SUBGROUP_METRICS)
    ci = None
    if n_boot > 0 and not small:
        res = metrics_with_ci(p, y, name, threshold, n_boot, alpha, seed, stratified, bins,
                              SUBGROUP_METRICS)["metrics"]
        ci = {m: res[m]["ci"] for m in SUBGROUP_METRICS}
    return {"group": label, "n": int(p.shape[0]), "small": small, "metrics": est, "ci": ci}


def subgroup_metrics(probs, targets, task, groups, threshold: float = 0.5, min_size: int = 20,
                     n_boot: int = 0, alpha: float = 0.05, seed: int = 0,
                     stratified: bool = True, bins: int = 15) -> dict:
    """AUC (macro), F1 (macro), sensitivity, specificity, ECE and ``n`` for every group.

    ``groups`` is a 1-D array with one label per sample (strings or numbers; bin continuous
    variables such as age beforehand). Metrics are those of ``metrics_with_ci`` (same definitions
    as ``per_class_curves``, ``threshold_sweep`` at ``threshold`` and ``calibration``), computed
    on the samples of each group.

    * Groups with fewer than ``min_size`` samples are returned with ``small = True`` and **no
      interval**, and are left out of the gap: their numbers are not reliable.
    * ``n_boot > 0`` adds the percentile bootstrap interval (``alpha``, resampling inside the
      group, see ``bootstrap.py``) of each metric under ``ci``; each group uses ``seed`` plus its
      position, so the result is deterministic. Cost: one ``metrics_with_ci`` per group.

    Returns::

        {"groups": [{"group", "n", "small", "metrics": {m: float}, "ci": {m: (lo, hi)} | None}],
         "overall": {same entry for all the samples, "group": None},
         "gap": {m: {"gap", "max", "min", "max_group", "min_group"}},
         "min_size", "n_boot", "alpha"}

    Groups are sorted by label. ``gap[m]`` = max - min of the metric over the non-small groups
    where it is defined (NaN, with ``None`` groups, if fewer than two). The gap is a descriptive
    difference of point estimates: it does not say whether it exceeds sampling noise (use the
    intervals).
    """
    name = task_name(task)
    if min_size < 1:
        raise ValueError("min_size must be >= 1")
    p, y = check_pair(probs, targets)
    g = _check_groups(groups, p.shape[0])
    entries = []
    for k, value in enumerate(np.unique(g)):
        rows = g == value
        entries.append(_entry(_label(value), p[rows], y[rows], name, threshold, bins, n_boot,
                              alpha, seed + k, stratified, int(rows.sum()) < min_size))
    overall = _entry(None, p, y, name, threshold, bins, n_boot, alpha, seed, stratified, False)
    return {"groups": entries, "overall": overall, "gap": _gaps(entries),
            "min_size": min_size, "n_boot": n_boot, "alpha": alpha}


def _gaps(entries) -> dict:
    out = {}
    for m in SUBGROUP_METRICS:
        valid = [(e["metrics"][m], e["group"]) for e in entries
                 if not e["small"] and np.isfinite(e["metrics"][m])]
        if len(valid) < 2:
            out[m] = {"gap": float("nan"), "max": float("nan"), "min": float("nan"),
                      "max_group": None, "min_group": None}
            continue
        hi, lo = max(valid, key=lambda t: t[0]), min(valid, key=lambda t: t[0])
        out[m] = {"gap": float(hi[0] - lo[0]), "max": float(hi[0]), "min": float(lo[0]),
                  "max_group": hi[1], "min_group": lo[1]}
    return out
