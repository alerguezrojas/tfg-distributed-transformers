"""Macro metrics against the decision threshold."""
import numpy as np

from src.analysis._common import check_pair, predict, task_name

#: 0.05, 0.10 ... 0.95 (wider than ``metrics.THRESHOLD_GRID``, which only searches the optimum).
DEFAULT_GRID = np.round(np.arange(1, 20) * 0.05, 2)

# Same epsilon as src/training/metrics.py, so the values coincide to float precision.
_EPS = 1e-8


def threshold_sweep(probs, targets, task, grid=None) -> dict:
    """Macro F1, precision, recall, sensitivity and specificity at each threshold.

    Semantics of ``src/training/metrics.py``: per-class columns, ``tp/(tp+fp+1e-8)`` etc., then
    the mean over classes (F1 = macro of the per-class F1, not the F1 of the macros); predictions
    are ``probs > threshold`` as in ``Task.predict``. ``sensitivity`` is the recall under its
    clinical name; ``specificity`` = mean of ``tn/(tn+fp+1e-8)``.

    Multiclass has no threshold (the prediction is the argmax): the result is a single point
    with ``thresholds = [nan]`` and ``grid`` is ignored.

    Returns arrays of length T (``thresholds``, ``f1``, ``precision``, ``recall``,
    ``sensitivity``, ``specificity``), the per-class versions as (T, C) arrays under
    ``per_class`` (same keys, minus ``thresholds``), and ``best`` = ``{"threshold", "f1"}``
    (first maximum of the macro F1; NaN threshold in multiclass).
    """
    name = task_name(task)
    p, y = check_pair(probs, targets)
    if name == "multiclass":
        grid = np.array([np.nan])
    else:
        grid = DEFAULT_GRID if grid is None else np.asarray(grid, dtype=np.float64).ravel()
    per = {k: [] for k in ("f1", "precision", "recall", "specificity")}
    for t in grid:
        pred = predict(p, name, t)
        tp = (pred & y).sum(axis=0).astype(np.float64)
        fp = (pred & ~y).sum(axis=0)
        fn = (~pred & y).sum(axis=0)
        tn = (~pred & ~y).sum(axis=0)
        prec = tp / (tp + fp + _EPS)
        rec = tp / (tp + fn + _EPS)
        per["precision"].append(prec)
        per["recall"].append(rec)
        per["f1"].append(2 * prec * rec / (prec + rec + _EPS))
        per["specificity"].append(tn / (tn + fp + _EPS))
    per = {k: np.array(v) for k, v in per.items()}          # (T, C)
    macro = {k: v.mean(axis=1) for k, v in per.items()}
    per["sensitivity"] = per["recall"]
    macro["sensitivity"] = macro["recall"]
    best = int(np.argmax(macro["f1"])) if len(grid) else 0
    return {
        "thresholds": grid,
        **macro,
        "per_class": per,
        "best": {"threshold": float(grid[best]), "f1": float(macro["f1"][best])},
    }
