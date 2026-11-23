"""Choosing an ensemble of runs on the **validation** split (pure NumPy, no I/O, no torch).

The ensemble is always the (weighted) mean of the runs' probabilities. The choice only looks at
validation probabilities; the test split is evaluated once, afterwards, with the chosen set.

* :func:`greedy_selection`: forward selection of Caruana et al. (2004, "Ensemble selection from
  libraries of models"): at each step add the run that raises the validation score of the mean the
  most, and stop when nothing improves.
* :func:`best_per_group`, :func:`topk_curve`, :func:`choose_k`: the best run of each group (e.g.
  architecture), ranked by validation score, and the k (number of groups) of best validation.
* :func:`average`: the mean of probabilities, the combination rule of both.

A score is ``score_fn(probs, targets) -> float`` (higher is better), e.g. a metric of
``src.analysis.compare.METRICS``. Ties are broken by run name, so the result is deterministic.
"""
import numpy as np


def average(probs_list, weights=None) -> np.ndarray:
    """Weighted mean of (N, C) probability arrays (equal weights by default)."""
    if not len(probs_list):
        raise ValueError("nothing to average")
    w = np.ones(len(probs_list)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(probs_list),) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative, one per array, with a positive sum")
    stack = np.stack([np.asarray(p, dtype=np.float64) for p in probs_list])
    return np.tensordot(w / w.sum(), stack, axes=1)


def greedy_selection(val_probs: dict, targets, score_fn, max_size: int = 8,
                     with_replacement: bool = True) -> tuple[list[str], list[tuple[str, float]]]:
    """Forward selection on validation. Returns ``(chosen, trace)``.

    ``chosen`` lists the run names (repeated when ``with_replacement``: a repeat is a larger
    weight); ``trace`` has the ``(name added, score of the ensemble so far)`` of every step. It
    stops at ``max_size`` or when no candidate strictly improves the score.
    """
    if not val_probs:
        raise ValueError("no candidate runs")
    if max_size < 1:
        raise ValueError("max_size must be >= 1")
    chosen: list[str] = []
    trace: list[tuple[str, float]] = []
    best = -np.inf
    while len(chosen) < max_size:
        step_name, step_score = None, -np.inf
        for name in sorted(val_probs):
            if not with_replacement and name in chosen:
                continue
            s = score_fn(average([val_probs[n] for n in chosen + [name]]), targets)
            if s > step_score:              # strict: the first name in order wins a tie
                step_name, step_score = name, float(s)
        if step_name is None or step_score <= best:
            break
        chosen.append(step_name)
        trace.append((step_name, step_score))
        best = step_score
    return chosen, trace


def best_per_group(val_scores: dict, group_of: dict) -> list[tuple[str, str, float]]:
    """Best run of each group: ``[(group, run, validation score)]``, best score first.

    ``group_of`` maps every run to its group; ties inside a group go to the smaller name, ties
    between groups to the smaller group name.
    """
    missing = [r for r in val_scores if r not in group_of]
    if missing:
        raise ValueError(f"runs without a group: {missing}")
    best: dict[str, tuple[str, float]] = {}
    for run in sorted(val_scores):
        g, s = group_of[run], float(val_scores[run])
        if g not in best or s > best[g][1]:
            best[g] = (run, s)
    return sorted(((g, r, s) for g, (r, s) in best.items()), key=lambda t: (-t[2], t[0]))


def topk_curve(val_probs: dict, targets, ranked: list[str], score_fn) -> list[tuple[int, float]]:
    """Validation score of the mean of the first k runs of ``ranked``, for k = 1..len(ranked)."""
    return [(k, float(score_fn(average([val_probs[n] for n in ranked[:k]]), targets)))
            for k in range(1, len(ranked) + 1)]


def choose_k(curve: list[tuple[int, float]]) -> int:
    """The k of the highest validation score; the smallest k on a tie."""
    if not curve:
        raise ValueError("empty curve")
    return max(curve, key=lambda t: (t[1], -t[0]))[0]
