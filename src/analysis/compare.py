"""Paired comparison of two groups of runs (e.g. 3 seeds of a variant A against 3 of a base B).

Pure NumPy, no I/O: it takes the per-sample probabilities that ``scripts/eval.py
--save-predictions`` writes (``probs`` (N, C), ``targets`` (N, C) one-hot) already loaded.

* :func:`run_metrics`: metrics of one run (and the main one per subgroup).
* :func:`paired_bootstrap`: the difference A - B of any statistic of the runs, with the **same**
  resampling indices for every run (so the runs' correlation, same test set, is kept), a 95 %
  percentile interval (Efron and Tibshirani, 1993) and ``P(d <= 0)``.
* :func:`compare_groups`: everything at once (per run, mean ± std, ensemble, bootstrap of the
  difference of the mean of the seeds, of the ensembles and, if the groups have the same size,
  per pair of seeds).

A metric is ``fn(probs, targets) -> float`` over (N, C) arrays, so a task with another shape
(multi-label) only needs its own entries in :data:`METRICS`; only the multiclass / ordinal ones
are defined and tested here.
"""
import numpy as np

from src.analysis.clinical import _qwk

#: Metrics that only make sense when the classes are ordered (a grade).
ORDINAL_METRICS = ("qwk", "acc_pm1")


def _grade(targets: np.ndarray) -> np.ndarray:
    return np.asarray(targets).argmax(axis=1)


def _qwk_metric(probs, targets) -> float:
    return _qwk(probs.argmax(axis=1), _grade(targets), probs.shape[1])


def _accuracy(probs, targets) -> float:
    return float((probs.argmax(axis=1) == _grade(targets)).mean())


def _acc_pm1(probs, targets) -> float:
    return float((np.abs(probs.argmax(axis=1) - _grade(targets)) <= 1).mean())


def rank_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """ROC AUC by ranks (Mann-Whitney U), ties get the average rank. NaN if a class is empty."""
    score, positive = np.asarray(score, dtype=np.float64), np.asarray(positive, dtype=bool)
    n1 = int(positive.sum())
    n0 = positive.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    # Average rank of every run of equal scores.
    start = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    end = np.append(start[1:], s.size)
    avg = (start + 1 + end) / 2.0
    ranks = np.empty(s.size)
    ranks[order] = np.repeat(avg, end - start)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _auc_ge1(probs, targets) -> float:
    """AUC of "class >= 1" with the score 1 - p(class 0) (PANDA: is there cancer?)."""
    return rank_auc(1.0 - probs[:, 0], _grade(targets) >= 1)


#: name -> (function, short label for the tables)
METRICS = {
    "qwk": (_qwk_metric, "QWK"),
    "accuracy": (_accuracy, "Exact."),
    "acc_pm1": (_acc_pm1, "±1 grado"),
    "auc_ge1": (_auc_ge1, "AUC clase ≥ 1"),
}


def default_metrics(ordinal: bool) -> list[str]:
    return [m for m in METRICS if ordinal or m not in ORDINAL_METRICS]


def default_primary(ordinal: bool) -> str:
    return "qwk" if ordinal else "accuracy"


def _check_metrics(metrics, ordinal: bool) -> None:
    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"unknown metric {m!r}; available: {list(METRICS)}")
        if m in ORDINAL_METRICS and not ordinal:
            raise ValueError(f"metric {m!r} needs an ordinal task")


def run_metrics(probs, targets, groups=None, metrics=None, primary=None, ordinal=True) -> dict:
    """``{metric: value}`` of one run plus ``"groups": {value: primary metric on that subgroup}``."""
    probs, targets = np.asarray(probs, dtype=np.float64), np.asarray(targets)
    metrics = list(metrics) if metrics is not None else default_metrics(ordinal)
    primary = primary or default_primary(ordinal)
    _check_metrics(metrics + [primary], ordinal)
    out = {m: METRICS[m][0](probs, targets) for m in metrics}
    out["groups"] = {}
    if groups is not None:
        groups = np.asarray(groups)
        for v in sorted(set(groups.tolist())):
            sel = groups == v
            out["groups"][str(v)] = METRICS[primary][0](probs[sel], targets[sel])
    return out


def mean_std(values) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1; NaN with a single value)."""
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")


def check_aligned(targets: list, groups: list | None = None) -> None:
    """All runs must come from the same test set, in the same order."""
    ref = np.asarray(targets[0])
    for i, t in enumerate(targets[1:], start=1):
        t = np.asarray(t)
        if t.shape != ref.shape or not np.array_equal(t, ref):
            raise ValueError(f"run {i} does not have the same targets as run 0 "
                             "(not the same test set or order)")
    if groups is not None:
        ref_g = np.asarray(groups[0])
        for i, g in enumerate(groups[1:], start=1):
            if np.asarray(g).shape != ref_g.shape or not np.array_equal(np.asarray(g), ref_g):
                raise ValueError(f"run {i} does not have the same subgroups as run 0")


def paired_bootstrap(stat_a, stat_b, n: int, n_boot: int = 2000, seed: int = 0,
                     indices: np.ndarray | None = None) -> dict:
    """Bootstrap of ``stat_a(idx) - stat_b(idx)`` over the same resampled sample indices.

    ``stat_*`` are callables taking an index array. Returns ``diff`` (on the full set), ``lo`` and
    ``hi`` (95 % percentile interval) and ``p`` = P(d <= 0) over the resamples. Pass ``indices``
    (``n_boot`` x ``n``) to reuse the same resamples across several calls.
    """
    if indices is None:
        indices = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    d = np.array([stat_a(i) - stat_b(i) for i in indices])
    lo, hi = np.percentile(d, [2.5, 97.5])
    full = np.arange(n)
    return {"diff": float(stat_a(full) - stat_b(full)), "lo": float(lo), "hi": float(hi),
            "p": float((d <= 0).mean())}


def compare_groups(probs_a, probs_b, targets, groups=None, metrics=None, primary=None,
                   ordinal=True, n_boot: int = 2000, seed: int = 0) -> dict:
    """Compare two groups of runs on the same test set (see the module docstring).

    ``probs_a`` / ``probs_b``: lists of (N, C) arrays; ``targets``: (N, C) shared; ``groups``:
    optional (N,) subgroup of each sample. Returns a dict with ``n``, ``metrics``, ``primary``,
    ``runs`` ({"a": [...], "b": [...]} of :func:`run_metrics`), ``summary`` ({"a"/"b": per
    metric (mean, std), "groups", "ensemble"}) and ``diff`` (``mean`` and ``ensemble``:
    {metric: bootstrap result}; ``pairs``: list of {metric: result}, empty if the sizes differ).
    """
    if not probs_a or not probs_b:
        raise ValueError("each group needs at least one run")
    probs_a = [np.asarray(p, dtype=np.float64) for p in probs_a]
    probs_b = [np.asarray(p, dtype=np.float64) for p in probs_b]
    targets = np.asarray(targets)
    for p in probs_a + probs_b:
        if p.shape != targets.shape:
            raise ValueError(f"probs {p.shape} and targets {targets.shape} must have the same shape")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    metrics = list(metrics) if metrics is not None else default_metrics(ordinal)
    primary = primary or default_primary(ordinal)
    _check_metrics(metrics + [primary], ordinal)
    n = targets.shape[0]
    runs = {k: [run_metrics(p, targets, groups, metrics, primary, ordinal) for p in ps]
            for k, ps in (("a", probs_a), ("b", probs_b))}
    ens = {"a": np.mean(probs_a, axis=0), "b": np.mean(probs_b, axis=0)}
    summary = {}
    for k in ("a", "b"):
        summary[k] = {m: mean_std([r[m] for r in runs[k]]) for m in metrics}
        names = runs[k][0]["groups"]
        summary[k]["groups"] = {g: mean_std([r["groups"][g] for r in runs[k]]) for g in names}
        summary[k]["ensemble"] = run_metrics(ens[k], targets, groups, metrics, primary, ordinal)

    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    fns = {m: METRICS[m][0] for m in metrics}

    def on(p, fn):
        return lambda i: fn(p[i], targets[i])

    def mean_of(ps, fn):
        return lambda i: float(np.mean([fn(p[i], targets[i]) for p in ps]))

    def boot(sa, sb):
        return paired_bootstrap(sa, sb, n, indices=idx)

    diff = {"mean": {m: boot(mean_of(probs_a, f), mean_of(probs_b, f)) for m, f in fns.items()},
            "ensemble": {m: boot(on(ens["a"], f), on(ens["b"], f)) for m, f in fns.items()},
            "pairs": []}
    if len(probs_a) == len(probs_b):
        diff["pairs"] = [{m: boot(on(pa, f), on(pb, f)) for m, f in fns.items()}
                         for pa, pb in zip(probs_a, probs_b)]
    return {"n": n, "n_boot": n_boot, "seed": seed, "metrics": metrics, "primary": primary,
            "runs": runs, "summary": summary, "diff": diff}
