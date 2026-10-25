"""Operating point: sensitivity, specificity, PPV and NPV at a threshold and by clinical criteria.

* Fixed threshold: :func:`operating_point`.
* Threshold that reaches a target sensitivity (e.g. 0.90 or 0.95) and the specificity it leaves:
  :func:`threshold_for_sensitivity` (the usual "rule-out" criterion).
* Youden's J = sensitivity + specificity - 1 (Youden 1950, "Index for rating diagnostic tests",
  Cancer 3(1):32-35): :func:`youden_threshold`.
* PPV and NPV depend on the prevalence (Altman and Bland 1994, "Diagnostic tests 2: predictive
  values", BMJ 309:102): by Bayes, from sensitivity and specificity, for the observed prevalence
  (default; then they equal TP/(TP+FP) and TN/(TN+FN)) or for the one of the target population.

Everything is per class (one-vs-rest in multiclass, where the thresholds act on the probability of
each class, not on the argmax). Unlike ``threshold_sweep`` there is no ``1e-8`` in the
denominators: an undefined ratio (class without positives, no predicted positives...) is NaN.
Thresholds found from the data are placed **between** two consecutive distinct scores (midpoint),
so they work the same with ``p > t`` (the rule of ``Task.predict``) and ``p >= t`` and are robust
to float16 rounding; the exact score of the cut is returned as ``score`` (positive = ``p >= score``).
A threshold fixed on validation should be applied to test, not re-estimated there.
"""
import numpy as np

from src.analysis._common import check_pair, predict, task_name
from src.analysis.curves import roc_curve

_TOL = 1e-12


def _ratio(num, den):
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _prevalence(y: np.ndarray, prevalence) -> np.ndarray:
    """(C,) prevalence: observed if ``prevalence`` is None, else a scalar or (C,) in (0, 1)."""
    n_cls = y.shape[1]
    if prevalence is None:
        return y.mean(axis=0) if len(y) else np.full(n_cls, np.nan)
    prev = np.broadcast_to(np.asarray(prevalence, dtype=np.float64), (n_cls,)).copy()
    if np.any((prev <= 0) | (prev >= 1)):
        raise ValueError("prevalence must be in (0, 1)")
    return prev


def predictive_values(sensitivity, specificity, prevalence) -> tuple[np.ndarray, np.ndarray]:
    """(PPV, NPV) by Bayes: PPV = se*pi / (se*pi + (1-sp)(1-pi)), NPV = sp(1-pi) /
    (sp(1-pi) + (1-se)*pi). Vectorised; NaN where the denominator is 0."""
    se, sp, pi = (np.asarray(v, dtype=np.float64) for v in (sensitivity, specificity, prevalence))
    ppv = _ratio(se * pi, se * pi + (1 - sp) * (1 - pi))
    npv = _ratio(sp * (1 - pi), sp * (1 - pi) + (1 - se) * pi)
    return ppv, npv


def _macro(d: dict, keys) -> dict:
    out = {}
    for k in keys:
        a = np.asarray(d[k], dtype=np.float64)
        a = a[~np.isnan(a)]
        out[k] = float(a.mean()) if a.size else float("nan")
    return out


def operating_point(probs, targets, task, threshold: float = 0.5, prevalence=None) -> dict:
    """Confusion counts, sensitivity, specificity, PPV and NPV at a fixed threshold, per class.

    Prediction = ``probs > threshold`` per class (argmax in multiclass, where ``threshold`` is
    unused: one-vs-rest counts of the argmax rule). ``prevalence``: None = the observed one of
    each class; a scalar or a (C,) array = the prevalence of another population (PPV/NPV by Bayes).

    Returns arrays of length C: ``tp, fp, fn, tn`` (int), ``sensitivity, specificity, ppv, npv,
    prevalence``, plus ``threshold`` and ``macro`` (mean over the classes where defined).
    """
    name = task_name(task)
    p, y = check_pair(probs, targets)
    pred = predict(p, name, threshold)
    tp, fp = (pred & y).sum(axis=0), (pred & ~y).sum(axis=0)
    fn, tn = (~pred & y).sum(axis=0), (~pred & ~y).sum(axis=0)
    se, sp = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    prev = _prevalence(y, prevalence)
    ppv, npv = predictive_values(se, sp, prev)
    out = {"threshold": float(threshold) if name != "multiclass" else float("nan"),
           "tp": tp, "fp": fp, "fn": fn, "tn": tn, "sensitivity": se, "specificity": sp,
           "ppv": ppv, "npv": npv, "prevalence": prev}
    out["macro"] = _macro(out, ("sensitivity", "specificity", "ppv", "npv"))
    return out


def _cut_threshold(thr: np.ndarray, i: int) -> float:
    """Threshold between ``thr[i]`` (positive) and the next lower distinct score."""
    if i + 1 < thr.size:
        return float((thr[i] + thr[i + 1]) / 2.0)
    return float(np.nextafter(thr[i], -np.inf))


def _from_roc(p, y, prevalence, choose) -> dict:
    """Per-class operating point where ``choose(fpr, tpr)`` returns the ROC index to use (>= 1)
    or None; shared by the sensitivity target and Youden."""
    n_cls = p.shape[1]
    keys = ("threshold", "score", "sensitivity", "specificity", "j")
    out = {k: np.full(n_cls, np.nan) for k in keys}
    for c in range(n_cls):
        fpr, tpr, thr, _ = roc_curve(p[:, c], y[:, c], max_points=None)
        if fpr.size == 0:
            continue
        i = choose(fpr, tpr)
        if i is None:
            continue
        out["threshold"][c] = _cut_threshold(thr, i)
        out["score"][c] = thr[i]
        out["sensitivity"][c] = tpr[i]
        out["specificity"][c] = 1.0 - fpr[i]
        out["j"][c] = tpr[i] - fpr[i]
    prev = _prevalence(y, prevalence)
    out["ppv"], out["npv"] = predictive_values(out["sensitivity"], out["specificity"], prev)
    out["prevalence"] = prev
    out["macro"] = _macro(out, ("sensitivity", "specificity", "ppv", "npv"))
    return out


def threshold_for_sensitivity(probs, targets, task, target: float = 0.90,
                              prevalence=None) -> dict:
    """Per class, the highest threshold whose sensitivity is at least ``target``, with the
    specificity (and PPV/NPV) that leaves. ``target`` in (0, 1].

    Thresholds come from the ROC points of the data (one per distinct score), so the achieved
    sensitivity is the smallest one on the data that is >= ``target`` (with few positives it can
    be well above). NaN for a class without positives or without negatives.

    Returns arrays of length C: ``threshold`` (midpoint, usable with ``>`` and ``>=``),
    ``score`` (positive = ``p >= score``), ``sensitivity``, ``specificity``, ``ppv``, ``npv``,
    ``prevalence``, ``j`` (Youden's J there), plus ``target`` and ``macro``.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError("target must be in (0, 1]")
    task_name(task)
    p, y = check_pair(probs, targets)

    def choose(fpr, tpr):
        hit = np.nonzero(tpr >= target - _TOL)[0]
        return int(hit[0]) if hit.size else None

    out = _from_roc(p, y, prevalence, choose)
    out["target"] = float(target)
    return out


def youden_threshold(probs, targets, task, prevalence=None) -> dict:
    """Per class, the threshold that maximises Youden's J = sensitivity + specificity - 1.

    The first maximum along the ROC (the highest threshold) is taken. The point "predict nothing"
    is excluded, so the threshold is always a real one even if J <= 0 (no discrimination).
    Same output as :func:`threshold_for_sensitivity` (``j`` is the maximised value).
    """
    task_name(task)
    p, y = check_pair(probs, targets)

    def choose(fpr, tpr):
        if fpr.size < 2:
            return None
        return int(np.argmax(tpr[1:] - fpr[1:])) + 1

    return _from_roc(p, y, prevalence, choose)


def operating_points(probs, targets, task, threshold: float = 0.5,
                     sensitivities=(0.90, 0.95), prevalence=None) -> dict:
    """All the criteria in one call (what the web table needs).

    ``{"fixed": operating_point(...), "sensitivity": [threshold_for_sensitivity(..., target=t)
    for each t], "youden": youden_threshold(...)}``.
    """
    return {
        "fixed": operating_point(probs, targets, task, threshold, prevalence),
        "sensitivity": [threshold_for_sensitivity(probs, targets, task, t, prevalence)
                        for t in sensitivities],
        "youden": youden_threshold(probs, targets, task, prevalence),
    }
