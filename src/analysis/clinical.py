"""Clinical views of a test evaluation: cut-offs of an ordinal grade and metrics per group.

* :func:`clinical_cutoffs`: for ordered classes (an ISUP grade, a severity), each cut ``k``
  turns the problem into "grade >= k" with the score ``P(grade >= k) = probs[:, k:].sum(1)`` and the
  hard rule ``argmax >= k``; sensitivity, specificity, PPV and NPV of that rule and the AUC of the
  score. In prostate biopsies k=1 reads "any cancer" and k=2 "clinically significant cancer".
* :func:`group_summary`: n, accuracy, macro F1, macro AUC (and QWK if ordinal) for every value of a
  grouping variable, in the same terms as the aggregate lines of ``eval.py``.

Pure NumPy, no I/O. Undefined ratios are NaN (no ``1e-8``), as in ``operating_point``.
"""
import numpy as np

from src.analysis._common import check_pair, predict, task_name
from src.analysis.curves import roc_curve
from src.analysis.operating_point import _ratio


def clinical_cutoffs(probs, targets, class_names=None) -> list[dict]:
    """Binary "grade >= k" metrics for k = 1..C-1 of an ordinal multiclass problem.

    ``probs`` and ``targets`` are (N, C) with the class columns **ordered** (targets one-hot or
    soft: the true grade is their argmax). Returns one dict per cut with ``k``, ``label``
    (``">= <class k>"``), ``n_pos``, ``n_neg``, ``tp, fp, fn, tn``, ``sensitivity``,
    ``specificity``, ``ppv``, ``npv`` (observed prevalence) and ``auc`` (of ``P(grade >= k)``).
    """
    p, y = check_pair(probs, targets)
    n_cls = p.shape[1]
    if n_cls < 2:
        raise ValueError("clinical_cutoffs needs at least two classes")
    names = list(class_names) if class_names is not None else []
    grade = y.argmax(axis=1)
    pred_grade = p.argmax(axis=1)
    out = []
    for k in range(1, n_cls):
        pos, pred = grade >= k, pred_grade >= k
        score = p[:, k:].sum(axis=1)
        tp, fp = int((pred & pos).sum()), int((pred & ~pos).sum())
        fn, tn = int((~pred & pos).sum()), int((~pred & ~pos).sum())
        name = str(names[k]) if k < len(names) else str(k)
        out.append({
            "k": k, "label": f">= {name}", "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": float(_ratio(tp, tp + fn)), "specificity": float(_ratio(tn, tn + fp)),
            "ppv": float(_ratio(tp, tp + fp)), "npv": float(_ratio(tn, tn + fn)),
            "auc": roc_curve(score, pos, max_points=None)[3],
        })
    return out


def _nanmean(a) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def _qwk(pred_grade: np.ndarray, grade: np.ndarray, n_cls: int) -> float:
    """Quadratic weighted kappa, same definition as ``metrics.quadratic_weighted_kappa``."""
    observed = np.zeros((n_cls, n_cls))
    np.add.at(observed, (grade, pred_grade), 1.0)
    total = observed.sum()
    if n_cls < 2 or total == 0:
        return float("nan")
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    g = np.arange(n_cls, dtype=np.float64)
    weights = (g[:, None] - g[None, :]) ** 2 / (n_cls - 1) ** 2
    denom = (weights * expected).sum()
    return float("nan") if denom <= 0 else float(1.0 - (weights * observed).sum() / denom)


def _metrics(p, y, name, threshold, ordinal) -> dict:
    pred = predict(p, name, threshold)
    tp, fp, fn = (pred & y).sum(axis=0), (pred & ~y).sum(axis=0), (~pred & y).sum(axis=0)
    prec, rec = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = _ratio(2 * prec * rec, prec + rec)
    if name == "multiclass":
        acc = float((p.argmax(axis=1) == y.argmax(axis=1)).mean())
    else:
        acc = float((pred == y).mean())
    aucs = [roc_curve(p[:, c], y[:, c], max_points=None)[3] for c in range(p.shape[1])]
    out = {"accuracy": acc, "f1": _nanmean(f1), "auc": _nanmean(aucs)}
    if ordinal:
        out["qwk"] = _qwk(p.argmax(axis=1), y.argmax(axis=1), p.shape[1])
    return out


def group_summary(probs, targets, task, groups, threshold: float = 0.5,
                  ordinal: bool = False) -> list[dict]:
    """Main metrics of every value of every grouping variable.

    ``groups`` is ``{variable: (N,) labels}``. Returns, per variable and value (sorted by value),
    ``{"variable", "group", "n", "accuracy", "f1", "auc"[, "qwk"]}``; F1 and AUC are macro means
    over the classes where they are defined (NaN otherwise); ``qwk`` only if ``ordinal`` (a
    multiclass task). Accuracy follows ``Task.accuracy`` (argmax match in multiclass, mean
    per-label agreement otherwise).
    """
    name = task_name(task)
    if ordinal and name != "multiclass":
        raise ValueError("ordinal groups need a multiclass task")
    p, y = check_pair(probs, targets)
    rows = []
    for var, values in groups.items():
        g = np.asarray(values)
        if g.ndim != 1 or g.shape[0] != p.shape[0]:
            raise ValueError(f"group {var!r} must be 1-D with {p.shape[0]} entries; got {g.shape}")
        for value in np.unique(g):
            sel = g == value
            label = value.item() if hasattr(value, "item") else value
            rows.append({"variable": var, "group": label, "n": int(sel.sum()),
                         **_metrics(p[sel], y[sel], name, threshold, ordinal)})
    return rows
