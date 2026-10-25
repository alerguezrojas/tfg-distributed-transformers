"""Calibration: ECE, MCE, Brier and the reliability diagram.

Binning and ECE/MCE as in Guo et al. (2017), "On Calibration of Modern Neural Networks"
(arXiv 1706.04599): equal-width bins on the confidence, ECE = sum_b (n_b / n) |acc_b - conf_b|
and MCE = max_b |acc_b - conf_b| over the non-empty bins.
"""
import numpy as np

from src.analysis._common import check_pair, task_name


def calibration(probs, targets, task, bins: int = 15) -> dict:
    """Calibration of the predictions.

    * Multiclass, **top-label**: one prediction per sample; confidence = max probability,
      correct = the argmax is the true class.
    * Multi-label / binary: every (sample, class) is a binary prediction; it predicts positive
      when ``p >= 0.5``, confidence = p if positive else 1 - p, correct = prediction == label.
      (Confidence therefore lies in [0.5, 1]: the lower bins stay empty.)

    Bins are equal-width on [0, 1], left-closed (the last one also holds confidence 1).
    Brier: multiclass = mean over samples of sum_c (p - y)^2; multi-label/binary = mean over
    elements of (p - y)^2.

    Returns ``{"ece", "mce", "brier", "n", "edges" (bins+1,), "bins"}`` where ``bins`` is a list
    of ``{"lower", "upper", "confidence", "accuracy", "count"}`` (confidence and accuracy are
    NaN in empty bins). ECE and MCE are NaN if there are no predictions.
    """
    name = task_name(task)
    if bins < 1:
        raise ValueError("bins must be >= 1")
    p, y = check_pair(probs, targets)
    if name == "multiclass":
        conf = p.max(axis=1)
        correct = p.argmax(axis=1) == y.argmax(axis=1)
        brier = float(((p - y) ** 2).sum(axis=1).mean()) if len(p) else float("nan")
    else:
        pos = p >= 0.5
        conf = np.where(pos, p, 1.0 - p).ravel()
        correct = (pos == y).ravel()
        brier = float(((p - y) ** 2).mean()) if p.size else float("nan")

    idx = np.minimum((conf * bins).astype(np.int64), bins - 1)
    count = np.bincount(idx, minlength=bins)
    sum_conf = np.bincount(idx, weights=conf, minlength=bins)
    sum_acc = np.bincount(idx, weights=correct.astype(np.float64), minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(count > 0, sum_conf / count, np.nan)
        mean_acc = np.where(count > 0, sum_acc / count, np.nan)
    gap = np.abs(mean_acc - mean_conf)
    n = int(count.sum())
    edges = np.linspace(0.0, 1.0, bins + 1)
    return {
        "ece": float(np.nansum(gap * count) / n) if n else float("nan"),
        "mce": float(np.nanmax(gap)) if n else float("nan"),
        "brier": brier,
        "n": n,
        "edges": edges,
        "bins": [{"lower": float(edges[b]), "upper": float(edges[b + 1]),
                  "confidence": float(mean_conf[b]), "accuracy": float(mean_acc[b]),
                  "count": int(count[b])} for b in range(bins)],
    }
