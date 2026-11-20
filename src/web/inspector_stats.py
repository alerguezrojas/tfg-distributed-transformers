"""Pure helpers of the clinical views of the Inspector (no Streamlit, no torch).

Bootstrap intervals for the Curves view, the operating point of the Clinical view, the
interpretation of the calibration slope/intercept and the subgroup tables. The numerics live in
``src/analysis`` (``metrics_with_ci``, ``operating_point``, ``subgroup_metrics``...); this module
reshapes them into what the page draws and adds the one computation that package does not have:
the interval of the PPV/NPV/F1/accuracy at a fixed threshold (``clinical_report``).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.analysis.bootstrap import metrics_with_ci
from src.analysis.curves import roc_curve
from src.analysis.operating_point import (
    operating_point, predictive_values, threshold_for_sensitivity, youden_threshold,
)

N_BOOT_CHOICES = (200, 1000, 2000)
DEFAULT_SEED = 0
ALPHA = 0.05
MACRO_METRICS = ("auc", "ap", "f1", "sensitivity", "specificity", "accuracy")
_NAN = float("nan")
_BLOCK_ELEMENTS = 4_000_000          # max elements of a (resamples x N) block


def ci_note(n_boot: int, alpha: float = ALPHA, seed: int = DEFAULT_SEED) -> str:
    """The sentence that must accompany every interval of the page."""
    return (f"{100 * (1 - alpha):.0f}% CI, percentile bootstrap, n={n_boot:,} resamples "
            f"(seed {seed}).")


def fmt_ci(estimate: float, lo: float, hi: float, pattern: str = "{:.3f}") -> str:
    """``0.912 [0.887, 0.934]``; ``—`` for an undefined estimate, no brackets without interval."""
    if estimate is None or not math.isfinite(estimate):
        return "—"
    out = pattern.format(estimate)
    if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
        out += f" [{pattern.format(lo)}, {pattern.format(hi)}]"
    return out


# ── Bootstrap of the standard metrics ───────────────────────────────────────────

def _strip(res: dict) -> dict[str, dict[str, float]]:
    return {m: {"estimate": v["estimate"], "lo": v["ci"][0], "hi": v["ci"][1],
                "n_valid": v["n_boot"]} for m, v in res["metrics"].items()}


def macro_ci(probs, targets, task: str, threshold: float = 0.5, n_boot: int = 1000,
             seed: int = DEFAULT_SEED, alpha: float = ALPHA,
             metrics: Sequence[str] = MACRO_METRICS) -> dict[str, dict[str, float]]:
    """``{metric: {"estimate", "lo", "hi", "n_valid"}}`` of the macro metrics (``metrics_with_ci``
    without the bootstrap distributions, so it is light to cache)."""
    return _strip(metrics_with_ci(probs, targets, task, threshold, n_boot, alpha, seed,
                                  metrics=tuple(metrics)))


def class_ci(probs, targets, task: str, threshold: float = 0.5, n_boot: int = 1000,
             seed: int = DEFAULT_SEED, alpha: float = ALPHA) -> list[dict[str, dict[str, float]]]:
    """Per-class intervals (one-vs-rest in multiclass): one ``macro_ci``-like dict per class.

    Each column is bootstrapped as a binary problem (stratified by its label). AUC and AP always;
    F1, sensitivity and specificity at ``threshold`` except in multiclass, where the per-class
    decision comes from the argmax of all the classes and cannot be resampled column by column.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    which = ("auc", "ap") if task == "multiclass" else ("auc", "ap", "f1", "sensitivity",
                                                        "specificity")
    return [macro_ci(p[:, [c]], y[:, [c]], "binary", threshold, n_boot, seed + c, alpha, which)
            for c in range(p.shape[1])]


def ci_column(cis: Sequence[dict[str, dict[str, float]]], metric: str) -> list[str]:
    """One formatted ``est [lo, hi]`` string per class for ``metric`` (``—`` if not computed)."""
    out = []
    for ci in cis:
        v = ci.get(metric)
        out.append("—" if v is None else fmt_ci(v["estimate"], v["lo"], v["hi"]))
    return out


# ── Operating point (Clinical view) ─────────────────────────────────────────────

CRITERIA = {"fixed": "Fixed threshold", "validation": "Run's validation-optimal threshold",
            "sensitivity": "Target sensitivity", "youden": "Youden's J"}
_COUNTS = ("tp", "fp", "fn", "tn")
POINT_METRICS = ("sensitivity", "specificity", "ppv", "npv", "f1", "accuracy")


def _ratio(num, den):
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _point_metrics(tp, fp, fn, tn, prevalence) -> dict[str, np.ndarray]:
    """Metrics from count arrays (scalars or per-resample vectors). ``prevalence`` None =
    observed in each (re)sample."""
    n = tp + fp + fn + tn
    se, sp = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    prev = _ratio(tp + fn, n) if prevalence is None else np.full_like(se, prevalence)
    ppv, npv = predictive_values(se, sp, prev)
    return {"sensitivity": se, "specificity": sp, "ppv": ppv, "npv": npv,
            "f1": _ratio(2 * tp, 2 * tp + fp + fn), "accuracy": _ratio(tp + tn, n)}


def _boot_counts(pred: np.ndarray, y: np.ndarray, n_boot: int, seed: int,
                 stratified: bool) -> np.ndarray:
    """(n_boot, 4) tp, fp, fn, tn of bootstrap resamples (counts drawn as multinomial weights).

    ``stratified`` keeps the number of positives and negatives of the data (used when the
    prevalence is imposed by the user); otherwise the prevalence varies from resample to resample.
    """
    rng = np.random.default_rng(seed)
    ind = np.stack([pred & y, pred & ~y, ~pred & y, ~pred & ~y], axis=1).astype(np.float64)
    groups = [np.flatnonzero(y), np.flatnonzero(~y)] if stratified else [np.arange(len(y))]
    block = max(1, min(n_boot, _BLOCK_ELEMENTS // max(len(y), 1)))
    out = np.empty((n_boot, 4))
    for start in range(0, n_boot, block):
        rows = min(block, n_boot - start)
        tot = np.zeros((rows, 4))
        for g in groups:
            w = rng.multinomial(g.size, np.full(g.size, 1.0 / g.size), size=rows)
            tot += w @ ind[g]
        out[start:start + rows] = tot
    return out


def _interval(values: np.ndarray, alpha: float) -> tuple[float, float]:
    ok = values[np.isfinite(values)]
    if ok.size == 0:
        return _NAN, _NAN
    lo, hi = np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def clinical_report(probs, targets, cls: int, criterion: str = "fixed", threshold: float = 0.5,
                    sens_target: float = 0.90, prevalence: float | None = None,
                    n_boot: int = 0, seed: int = DEFAULT_SEED, alpha: float = ALPHA,
                    max_points: int = 600) -> dict[str, Any]:
    """Operating point of class ``cls`` (one-vs-rest in multiclass) with intervals.

    ``criterion``: ``fixed`` and ``validation`` use ``threshold`` (the second is just the run's
    own threshold, chosen by the caller); ``sensitivity`` takes the highest threshold with at
    least ``sens_target`` sensitivity; ``youden`` maximises sensitivity + specificity - 1. The
    prediction is ``p > threshold`` on the class probability. ``prevalence`` (0-1) replaces the
    observed one in the PPV/NPV (Bayes); ``None`` keeps the observed.

    ``n_boot > 0`` adds percentile intervals of sensitivity, specificity, PPV, NPV, F1 and
    accuracy at that threshold. They are *conditional on the threshold*: when it was chosen on
    these same data (``sensitivity``/``youden``) they ignore the uncertainty of the choice. With
    an imposed prevalence the resampling is stratified by label.

    Returns ``{"ok": False, "reason"}`` if the class has no positives or no negatives; otherwise
    ``{"ok": True, "threshold", "counts", "metrics": {name: {"estimate", "lo", "hi"}},
    "roc": {"fpr", "tpr", "auc"}, "point": (fpr, tpr), "prevalence": {"observed", "used"}, "n",
    "n_boot", "seed", "criterion"}``.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; available: {list(CRITERIA)}")
    if prevalence is not None and not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be in (0, 1)")
    p = np.asarray(probs, dtype=np.float64)[:, [cls]]
    y = np.asarray(targets, dtype=np.float64)[:, [cls]] > 0.5
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return {"ok": False, "reason": "The class has no positive or no negative samples in "
                                       "this data, so it has no operating point."}
    if criterion == "sensitivity":
        thr = float(threshold_for_sensitivity(p, y, "binary", sens_target)["threshold"][0])
    elif criterion == "youden":
        thr = float(youden_threshold(p, y, "binary")["threshold"][0])
    else:
        thr = float(threshold)
    op = operating_point(p, y, "binary", thr, prevalence)
    counts = {k: int(op[k][0]) for k in _COUNTS}
    est = _point_metrics(*(np.float64(counts[k]) for k in _COUNTS), prevalence)
    metrics = {m: {"estimate": float(est[m]), "lo": _NAN, "hi": _NAN} for m in POINT_METRICS}
    if n_boot > 0:
        pred = (p > thr)[:, 0]
        c = _boot_counts(pred, y[:, 0], n_boot, seed, stratified=prevalence is not None)
        boot = _point_metrics(c[:, 0], c[:, 1], c[:, 2], c[:, 3], prevalence)
        for m in POINT_METRICS:
            metrics[m]["lo"], metrics[m]["hi"] = _interval(boot[m], alpha)
    fpr, tpr, _, auc = roc_curve(p[:, 0], y[:, 0], max_points=max_points)
    observed = float(y.mean())
    return {"ok": True, "threshold": thr, "counts": counts, "metrics": metrics,
            "roc": {"fpr": fpr, "tpr": tpr, "auc": auc},
            "point": (1.0 - metrics["specificity"]["estimate"], metrics["sensitivity"]["estimate"]),
            "prevalence": {"observed": observed, "used": observed if prevalence is None
                           else float(prevalence)},
            "n": int(len(y)), "n_boot": int(n_boot), "seed": seed, "criterion": criterion}


def confusion_2x2(counts: dict[str, int]) -> pd.DataFrame:
    """Rows = actual, columns = predicted."""
    return pd.DataFrame(
        [[counts["tp"], counts["fn"]], [counts["fp"], counts["tn"]]],
        index=pd.Index(["Actual positive", "Actual negative"]),
        columns=["Predicted positive", "Predicted negative"])


def point_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Table ``Metric, Value [95% CI]`` of a ``clinical_report``."""
    names = {"sensitivity": "Sensitivity (recall)", "specificity": "Specificity",
             "ppv": "PPV (precision)", "npv": "NPV", "f1": "F1", "accuracy": "Accuracy"}
    m = report["metrics"]
    return pd.DataFrame({"Metric": [names[k] for k in POINT_METRICS],
                         "Value": [fmt_ci(m[k]["estimate"], m[k]["lo"], m[k]["hi"])
                                   for k in POINT_METRICS]})


# ── Calibration ─────────────────────────────────────────────────────────────────

_SLOPE_TOL = 0.1       # |slope - 1| treated as "close to 1" when there is no standard error


def interpret_slope(slope: float, se: float | None = None) -> tuple[str, str]:
    """(level, text) for the calibration slope. ``over`` / ``under`` when the Wald 95% interval
    (or, without a standard error, a ±0.1 band) excludes 1; ``ok`` otherwise; ``none`` if NaN."""
    if slope is None or not math.isfinite(slope):
        return "none", "The calibration slope is undefined (one outcome value or no fit)."
    half = 1.96 * se if se is not None and math.isfinite(se) else _SLOPE_TOL
    if slope + half < 1.0:
        return "over", (f"Slope {slope:.2f} < 1: overconfident. Predictions are too extreme "
                        "(high probabilities too high, low ones too low).")
    if slope - half > 1.0:
        return "under", (f"Slope {slope:.2f} > 1: underconfident. Predictions are too moderate; "
                         "the model could be more decisive.")
    return "ok", f"Slope {slope:.2f}: compatible with 1 (well calibrated in spread)."


def interpret_citl(intercept: float, tol: float = 0.1) -> str:
    """Text for the calibration-in-the-large (intercept with the slope fixed at 1)."""
    if intercept is None or not math.isfinite(intercept):
        return "Intercept undefined."
    if abs(intercept) < tol:
        return f"Intercept {intercept:+.2f}: close to 0 (no systematic bias in the risk)."
    if intercept > 0:
        return (f"Intercept {intercept:+.2f} > 0: the event happens more often than predicted "
                "(risk underestimated).")
    return (f"Intercept {intercept:+.2f} < 0: the event happens less often than predicted "
            "(risk overestimated).")


def calibration_class_table(classes: Sequence[str], per_class: dict, stats: dict,
                            ace: Sequence[float] | None = None) -> pd.DataFrame:
    """Per-class calibration: positives, ECE, MCE, Brier, adaptive ECE, slope and intercept."""
    fits = stats["per_class"]
    cols = {"Class": list(classes), "Positives": np.asarray(per_class["n_pos"]),
            "ECE": per_class["ece"], "MCE": per_class["mce"], "Brier": per_class["brier"]}
    if ace is not None:
        cols["ACE (equal mass)"] = np.asarray(ace, dtype=float)
    cols["Slope"] = [f["slope"] for f in fits]
    cols["Intercept (CITL)"] = [f["intercept_citl"] for f in fits]
    return pd.DataFrame(cols)


# ── Subgroups ───────────────────────────────────────────────────────────────────

SUBGROUP_LABELS = {"auc": "AUC", "f1": "F1", "sensitivity": "Sensitivity",
                   "specificity": "Specificity", "ece": "ECE"}


def bin_groups(values, n_bins: int = 4, max_unique: int = 12) -> np.ndarray:
    """Group labels (strings) from a variable: categorical values stay as they are; a numeric
    variable with more than ``max_unique`` distinct values (age, say) is cut in ``n_bins``
    quantile bands ``Q1 [lo, hi]``... Missing values (NaN) become the label ``missing``."""
    a = np.asarray(values)
    if a.dtype.kind in "iuf":
        finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
        if np.unique(finite).size > max_unique:
            edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)))
            idx = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(edges) - 2)
            labels = np.array([f"Q{i + 1} [{edges[i]:g}, {edges[i + 1]:g}]"
                               for i in range(len(edges) - 1)])
            out = labels[idx].astype(object)
            if a.dtype.kind == "f":
                out[~np.isfinite(a)] = "missing"
            return out.astype(str)
        if a.dtype.kind == "f":
            return np.where(np.isfinite(a), a.astype(str), "missing")
    return a.astype(str)


def subgroup_frame(result: dict) -> pd.DataFrame:
    """One row per group (plus ``All samples``): ``n`` and every metric as ``est [lo, hi]``."""
    rows = []
    for e in [{**result["overall"], "group": "All samples"}, *result["groups"]]:
        row: dict[str, Any] = {"Group": str(e["group"]),
                               "n": f"{e['n']} (small)" if e["small"] else str(e["n"])}
        for m, label in SUBGROUP_LABELS.items():
            lo, hi = (e["ci"][m] if e["ci"] else (_NAN, _NAN))
            row[label] = fmt_ci(e["metrics"][m], lo, hi)
        rows.append(row)
    return pd.DataFrame(rows)


def _s(v) -> str:
    return "—" if v is None else str(v)


def gap_frame(result: dict) -> pd.DataFrame:
    """Max-min gap of every metric with the groups that set it (NaN gap without two groups)."""
    rows = []
    for m, label in SUBGROUP_LABELS.items():
        g = result["gap"][m]
        rows.append({"Metric": label, "Gap (max - min)": g["gap"], "Best group": _s(g["max_group"]),
                     "Best": g["max"], "Worst group": _s(g["min_group"]), "Worst": g["min"]})
    return pd.DataFrame(rows)
