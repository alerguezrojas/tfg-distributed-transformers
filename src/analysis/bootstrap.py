"""Bootstrap confidence intervals (percentile method) for any metric and for the standard set.

Efron (1979), "Bootstrap methods: another look at the jackknife", Annals of Statistics 7(1):
1-26; Efron and Tibshirani (1993), "An Introduction to the Bootstrap". The percentile interval
is the (alpha/2, 1 - alpha/2) quantiles of the metric over resamples of the samples (rows) drawn
with replacement. It is simple and honest for the sizes of a validation set; with very few
samples (tens) or a metric at its bound (AUC near 1) it is optimistic: read it as a guide.

Two entry points:

* :func:`bootstrap_ci`: any ``metric_fn(probs, targets) -> float``; one call per resample.
* :func:`metrics_with_ci`: AUC, AP, F1, sensitivity, specificity, accuracy, ECE and Brier. It
  does **not** call the metric functions once per resample: a resample is a vector of integer
  weights (how many times each sample is drawn), and every metric is a weighted sum, so a block
  of resamples is computed at once (matrix products and cumulative sums; the sort for AUC/AP is
  done once). It matches the metric functions of this package exactly (tested resample by
  resample). Cost: 1000 resamples of 10 000 x 10 take a few seconds on a CPU core.
"""
import numpy as np

from src.analysis._common import check_pair, predict, task_name
from src.analysis.calibration import calibration
from src.analysis.curves import pr_curve, roc_curve
from src.analysis.thresholds import threshold_sweep

METRIC_NAMES = ("auc", "ap", "f1", "sensitivity", "specificity", "accuracy", "ece", "brier")

# Same epsilon as thresholds.py / training/metrics.py.
_EPS = 1e-8
# Max elements of a (resamples x N) block: bounds the memory of the vectorised path (~32 MB each).
_BLOCK_ELEMENTS = 4_000_000


# --------------------------------------------------------------------------- helpers

def _check_args(n_boot: int, alpha: float) -> None:
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _strata(y: np.ndarray, task: str | None, stratified: bool) -> list[np.ndarray] | None:
    """Index groups that are resampled separately (the class counts stay fixed), or None.

    Binary: the label; multiclass: the true class. Multi-label has no single class per sample,
    so it is resampled as a whole. With ``task=None`` it is inferred from the targets: one
    column = binary, exactly one positive per row = multiclass, otherwise multi-label.
    """
    if not stratified or y.shape[0] == 0:
        return None
    if task is None:
        if y.shape[1] == 1:
            task = "binary"
        elif np.all(y.sum(axis=1) == 1):
            task = "multiclass"
        else:
            task = "multilabel"
    if task == "binary":
        lab = y[:, 0].astype(np.int64)
    elif task == "multiclass":
        lab = y.argmax(axis=1)
    else:
        return None
    return [np.nonzero(lab == k)[0] for k in np.unique(lab)]


def _draw_indices(rng, n: int, strata) -> np.ndarray:
    if strata is None:
        return rng.integers(0, n, n)
    return np.concatenate([s[rng.integers(0, s.size, s.size)] for s in strata])


def _draw_weights(rng, n: int, strata, rows: int) -> np.ndarray:
    """(rows, n) float64 counts: how many times each sample is drawn in each resample."""
    if strata is None:
        return rng.multinomial(n, np.full(n, 1.0 / n), size=rows).astype(np.float64)
    w = np.zeros((rows, n))
    for s in strata:
        w[:, s] = rng.multinomial(s.size, np.full(s.size, 1.0 / s.size), size=rows)
    return w


def _interval(values: np.ndarray, alpha: float) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _summary(estimate: float, values: np.ndarray, n_requested: int, alpha: float) -> dict:
    ok = values[np.isfinite(values)]
    return {"estimate": float(estimate), "ci": _interval(ok, alpha), "n_boot": int(ok.size),
            "n_degenerate": int(n_requested - ok.size), "values": ok}


# --------------------------------------------------------------------------- generic

def bootstrap_ci(metric_fn, probs, targets, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0, stratified: bool = True, task=None) -> dict:
    """Percentile bootstrap interval of ``metric_fn(probs, targets) -> float``.

    Resamples the **samples** (rows) with replacement; ``probs`` and ``targets`` are float64
    (N, C) (targets 0/1). With ``stratified`` the resampling is done inside each class (binary:
    the label; multiclass: the true class), so every resample keeps the class counts; multi-label
    has no single class per sample and is resampled as a whole. ``task`` (name or ``Task``) only
    selects the strata; if omitted it is inferred from the targets (one column = binary, one
    positive per row = multiclass, else multi-label).

    A resample is *degenerate* and discarded when the metric is not finite (NaN/inf, e.g. an AUC
    with a single class) or raises ``ValueError``. The point estimate is the metric on the full
    data (an error there is not swallowed). Deterministic for a given ``seed``.

    Returns ``{"estimate", "ci": (lo, hi), "n_boot" (valid resamples), "n_degenerate",
    "alpha", "values" (the valid bootstrap distribution)}``; the interval is NaN if no resample
    is valid. Cost: ``n_boot`` calls of ``metric_fn``.
    """
    _check_args(n_boot, alpha)
    p, y = check_pair(probs, targets)
    y = y.astype(np.float64)
    n = p.shape[0]
    if n == 0:
        return {"estimate": float("nan"), "ci": (float("nan"), float("nan")), "n_boot": 0,
                "n_degenerate": n_boot, "alpha": alpha, "values": np.empty(0)}
    name = None if task is None else task_name(task)
    strata = _strata(y > 0.5, name, stratified)
    rng = np.random.default_rng(seed)
    estimate = float(metric_fn(p, y))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = _draw_indices(rng, n, strata)
        try:
            vals[b] = float(metric_fn(p[idx], y[idx]))
        except ValueError:
            vals[b] = np.nan
    out = _summary(estimate, vals, n_boot, alpha)
    out["alpha"] = alpha
    return out


# --------------------------------------------------------------------------- standard metrics

def _sample_accuracy(p, y, name, threshold) -> np.ndarray:
    """Per-sample accuracy with the semantics of ``Task.accuracy`` (1/0 in multiclass, the
    fraction of correct labels in multi-label / binary)."""
    pred = predict(p, name, threshold)
    if name == "multiclass":
        return (pred.argmax(axis=1) == y.argmax(axis=1)).astype(np.float64)
    return (pred == y).mean(axis=1)


def point_estimates(p, y, name, threshold, bins, which) -> dict:
    """Point estimates reusing the package functions (the reference of the bootstrap)."""
    out = {}
    if "auc" in which or "ap" in which:
        aucs = [roc_curve(p[:, c], y[:, c], max_points=2)[3] for c in range(p.shape[1])]
        aps = [pr_curve(p[:, c], y[:, c], max_points=2)[3] for c in range(p.shape[1])]
        out["auc"], out["ap"] = _nanmean(aucs), _nanmean(aps)
    if {"f1", "sensitivity", "specificity"} & set(which):
        sweep = threshold_sweep(p, y, name, grid=[threshold])
        for k in ("f1", "sensitivity", "specificity"):
            out[k] = float(sweep[k][0])
    if "accuracy" in which:
        out["accuracy"] = float(_sample_accuracy(p, y, name, threshold).mean()) \
            if len(p) else float("nan")
    if "ece" in which or "brier" in which:
        cal = calibration(p, y, name, bins=bins)
        out["ece"], out["brier"] = cal["ece"], cal["brier"]
    return {k: out[k] for k in which}


def _nanmean(values) -> float:
    a = np.asarray(values, dtype=np.float64)
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


class _Ranked:
    """Per-class sort for the weighted AUC/AP, done once for all the resamples."""

    def __init__(self, scores: np.ndarray, target: np.ndarray):
        self.order = np.argsort(-scores, kind="mergesort")
        s = scores[self.order]
        self.pos = target[self.order].astype(np.float64)
        self.ends = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]

    def auc_ap(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(auc, ap) of every row of ``w``; NaN where the resample lacks a class."""
        ws = w[:, self.order]
        tps = np.cumsum(ws * self.pos, axis=1)[:, self.ends]
        fps = np.cumsum(ws, axis=1)[:, self.ends] - tps
        n_pos, n_neg = tps[:, -1:], fps[:, -1:]
        valid = ((n_pos > 0) & (n_neg > 0))[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            tpr, fpr = tps / n_pos, fps / n_neg
            tot = tps + fps
            prec = np.where(tot > 0, tps / np.where(tot > 0, tot, 1.0), 0.0)
        zero = np.zeros((w.shape[0], 1))
        tpr_prev = np.concatenate([zero, tpr[:, :-1]], axis=1)
        auc = (np.diff(fpr, axis=1, prepend=0.0) * (tpr + tpr_prev) / 2).sum(axis=1)
        ap = (np.diff(tpr, axis=1, prepend=0.0) * prec).sum(axis=1)
        return np.where(valid, auc, np.nan), np.where(valid, ap, np.nan)


class _Weighted:
    """Everything the weighted metrics need, computed once per dataset."""

    def __init__(self, p, y, name, threshold, bins, which):
        self.which, self.name, self.n_cls = which, name, p.shape[1]
        if "auc" in which or "ap" in which:
            self.ranked = [_Ranked(p[:, c], y[:, c]) for c in range(p.shape[1])]
        if {"f1", "sensitivity", "specificity"} & set(which):
            pred = predict(p, name, threshold)
            self.counts = np.concatenate(
                [pred & y, pred & ~y, ~pred & y, ~pred & ~y], axis=1).astype(np.float64)
        if "accuracy" in which:
            self.acc = _sample_accuracy(p, y, name, threshold)
        if "ece" in which or "brier" in which:
            self._calibration_terms(p, y, bins)

    def _calibration_terms(self, p, y, bins):
        """Per-sample terms of ECE (sum of acc - conf per bin) and Brier, as in ``calibration``."""
        n = p.shape[0]
        if self.name == "multiclass":
            conf = p.max(axis=1)[:, None]
            correct = (p.argmax(axis=1) == y.argmax(axis=1))[:, None]
            self.brier = ((p - y) ** 2).sum(axis=1)
            self.per_sample = 1
        else:
            pos = p >= 0.5
            conf = np.where(pos, p, 1.0 - p)
            correct = pos == y
            self.brier = ((p - y) ** 2).sum(axis=1)
            self.per_sample = p.shape[1]
        idx = np.minimum((conf * bins).astype(np.int64), bins - 1)
        diff = correct.astype(np.float64) - conf
        self.cal = np.zeros((n, bins))
        rows = np.arange(n)
        for c in range(conf.shape[1]):
            self.cal[rows, idx[:, c]] += diff[:, c]

    def compute(self, w: np.ndarray) -> dict:
        """Metrics of every resample (row of ``w``) -> {name: (rows,) array}."""
        out, which, n_w = {}, self.which, w.sum(axis=1)
        if "auc" in which or "ap" in which:
            res = [r.auc_ap(w) for r in self.ranked]
            for key, k in (("auc", 0), ("ap", 1)):
                stack = np.stack([r[k] for r in res], axis=1)
                defined = ~np.isnan(stack)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.where(defined, stack, 0.0).sum(axis=1) / defined.sum(axis=1)
                out[key] = mean            # 0/0 -> NaN: no class defined
        if {"f1", "sensitivity", "specificity"} & set(which):
            c = self.n_cls
            tot = w @ self.counts
            tp, fp, fn, tn = tot[:, :c], tot[:, c:2 * c], tot[:, 2 * c:3 * c], tot[:, 3 * c:]
            prec, rec = tp / (tp + fp + _EPS), tp / (tp + fn + _EPS)
            out["f1"] = (2 * prec * rec / (prec + rec + _EPS)).mean(axis=1)
            out["sensitivity"] = rec.mean(axis=1)
            out["specificity"] = (tn / (tn + fp + _EPS)).mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            if "accuracy" in which:
                out["accuracy"] = (w @ self.acc) / n_w
            if "ece" in which:
                out["ece"] = np.abs(w @ self.cal).sum(axis=1) / (n_w * self.per_sample)
            if "brier" in which:
                out["brier"] = (w @ self.brier) / (n_w * self.per_sample)
        return {k: out[k] for k in which}


def metrics_with_ci(probs, targets, task, threshold: float = 0.5, n_boot: int = 1000,
                    alpha: float = 0.05, seed: int = 0, stratified: bool = True,
                    bins: int = 15, metrics=None) -> dict:
    """Point estimate and percentile bootstrap interval of the standard metrics.

    ``metrics`` selects a subset of :data:`METRIC_NAMES` (default all):

    * ``auc``, ``ap``: macro over the classes with positives and negatives (one-vs-rest in
      multiclass), as ``per_class_curves``.
    * ``f1``, ``sensitivity``, ``specificity``: macro at ``threshold`` (argmax in multiclass),
      as ``threshold_sweep``.
    * ``accuracy``: as ``Task.accuracy`` (sample-averaged label accuracy; exact class in multiclass).
    * ``ece``, ``brier``: as ``calibration`` (``bins`` equal-width bins).

    The estimates come from the package functions; the resamples from an equivalent weighted
    computation (see the module docstring), so 1000 resamples of 10 000 x C are a matter of
    seconds. Resampling and degenerate resamples as in :func:`bootstrap_ci` (a degenerate one is
    only possible for AUC/AP, if no class keeps positives and negatives).

    Returns ``{"n", "alpha", "n_boot" (requested), "threshold", "seed", "metrics": {name:
    {"estimate", "ci": (lo, hi), "n_boot" (valid), "n_degenerate", "values"}}}``.
    """
    name = task_name(task)
    _check_args(n_boot, alpha)
    which = tuple(METRIC_NAMES if metrics is None else metrics)
    unknown = [m for m in which if m not in METRIC_NAMES]
    if unknown or not which:
        raise ValueError(f"unknown metrics {unknown}; available: {list(METRIC_NAMES)}")
    p, y = check_pair(probs, targets)
    n = p.shape[0]
    head = {"n": n, "alpha": alpha, "n_boot": n_boot, "threshold": threshold, "seed": seed}
    if n == 0:
        nan = float("nan")
        return {**head, "metrics": {m: {"estimate": nan, "ci": (nan, nan), "n_boot": 0,
                                        "n_degenerate": n_boot, "values": np.empty(0)}
                                    for m in which}}
    est = point_estimates(p, y, name, threshold, bins, which)
    weighted = _Weighted(p, y, name, threshold, bins, which)
    strata = _strata(y, name, stratified)
    rng = np.random.default_rng(seed)
    block = max(1, min(n_boot, _BLOCK_ELEMENTS // n))
    vals = {m: [] for m in which}
    for start in range(0, n_boot, block):
        res = weighted.compute(_draw_weights(rng, n, strata, min(block, n_boot - start)))
        for m in which:
            vals[m].append(res[m])
    return {**head, "metrics": {m: _summary(est[m], np.concatenate(vals[m]), n_boot, alpha)
                                for m in which}}
