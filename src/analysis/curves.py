"""ROC and precision-recall curves, per class and aggregated.

Definitions follow the standard ones (scikit-learn's): one point per distinct score, AUC by the
trapezoid rule and average precision as the step sum  AP = sum (R_n - R_{n-1}) * P_n  (no
interpolation). A class without positives or without negatives has no curve: empty arrays and
NaN scores, never an exception. The scores are computed on the full curve; ``max_points`` only
thins the returned points (for plotting), always keeping the first and the last.
"""
import numpy as np

from src.analysis._common import check_pair

MAX_POINTS = 1000


def _clf_curve(scores: np.ndarray, target: np.ndarray):
    """(fps, tps, thresholds) at each distinct score, by decreasing threshold."""
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = target[order].astype(np.float64)
    last_of_group = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]
    tps = np.cumsum(y)[last_of_group]
    fps = 1 + last_of_group - tps
    return fps, tps, s[last_of_group]


def _thin(n: int, max_points: int | None) -> np.ndarray | slice:
    """Indices of at most ``max_points`` evenly spread points, ends included."""
    if max_points is None or n <= max_points:
        return slice(None)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def _prepare(scores, target):
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(target).ravel() > 0.5
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and target {y.shape} must have the same length")
    n_pos = int(y.sum())
    return s, y, n_pos, y.size - n_pos


def roc_curve(scores, target, max_points: int | None = MAX_POINTS):
    """ROC of one binary column -> ``(fpr, tpr, thresholds, auc)``.

    The curve starts at (0, 0) with threshold ``inf``; point i is "score >= thresholds[i]".
    Without positives or without negatives: three empty arrays and ``auc = nan``.
    """
    s, y, n_pos, n_neg = _prepare(scores, target)
    empty = np.empty(0)
    if n_pos == 0 or n_neg == 0:
        return empty, empty, empty, float("nan")
    fps, tps, thr = _clf_curve(s, y)
    fpr = np.r_[0.0, fps / n_neg]
    tpr = np.r_[0.0, tps / n_pos]
    thr = np.r_[np.inf, thr]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    keep = _thin(fpr.size, max_points)
    return fpr[keep], tpr[keep], thr[keep], auc


def pr_curve(scores, target, max_points: int | None = MAX_POINTS):
    """Precision-recall of one binary column -> ``(precision, recall, thresholds, ap)``.

    One point per distinct score, by decreasing threshold (recall non-decreasing), the three
    arrays aligned; point i is "score >= thresholds[i]". There is no artificial (recall 0,
    precision 1) point: prepend it to plot. Without positives or without negatives: empty
    arrays and ``ap = nan`` (the same rule as the ROC, so the macro averages agree).
    """
    s, y, n_pos, n_neg = _prepare(scores, target)
    empty = np.empty(0)
    if n_pos == 0 or n_neg == 0:
        return empty, empty, empty, float("nan")
    fps, tps, thr = _clf_curve(s, y)
    precision = tps / (tps + fps)
    recall = tps / n_pos
    ap = float(np.sum(np.diff(recall, prepend=0.0) * precision))
    keep = _thin(precision.size, max_points)
    return precision[keep], recall[keep], thr[keep], ap


def per_class_curves(probs, targets, max_points: int | None = MAX_POINTS) -> dict:
    """ROC and PR of every class (one-vs-rest in multiclass) plus macro and micro summaries.

    Returns::

        {"classes": [{"roc": {"fpr", "tpr", "thresholds", "auc"},
                      "pr":  {"precision", "recall", "thresholds", "ap"}}, ...],
         "auc": (C,) array, "ap": (C,) array,          # NaN where undefined
         "macro": {"auc", "ap"},    # mean over the classes with positives and negatives
         "micro": {"roc": {...}, "pr": {...}}}         # all (sample, class) pairs flattened

    NaN (never an exception) for a class without positives or negatives; the macro is NaN only
    if no class is defined.
    """
    p, y = check_pair(probs, targets)
    classes, aucs, aps = [], [], []
    for c in range(p.shape[1]):
        fpr, tpr, rt, auc = roc_curve(p[:, c], y[:, c], max_points)
        prec, rec, pt, ap = pr_curve(p[:, c], y[:, c], max_points)
        classes.append({"roc": {"fpr": fpr, "tpr": tpr, "thresholds": rt, "auc": auc},
                        "pr": {"precision": prec, "recall": rec, "thresholds": pt, "ap": ap}})
        aucs.append(auc)
        aps.append(ap)
    aucs, aps = np.array(aucs), np.array(aps)
    fpr, tpr, rt, micro_auc = roc_curve(p.ravel(), y.ravel(), max_points)
    prec, rec, pt, micro_ap = pr_curve(p.ravel(), y.ravel(), max_points)
    return {
        "classes": classes,
        "auc": aucs,
        "ap": aps,
        "macro": {"auc": _nanmean(aucs), "ap": _nanmean(aps)},
        "micro": {"roc": {"fpr": fpr, "tpr": tpr, "thresholds": rt, "auc": micro_auc},
                  "pr": {"precision": prec, "recall": rec, "thresholds": pt, "ap": micro_ap}},
    }


def _nanmean(a: np.ndarray) -> float:
    ok = a[~np.isnan(a)]
    return float(ok.mean()) if ok.size else float("nan")
