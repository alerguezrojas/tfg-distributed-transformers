"""Calibration: ECE, MCE, Brier, the reliability diagram, slope/intercept and class-wise errors.

Binning and ECE/MCE as in Guo et al. (2017), "On Calibration of Modern Neural Networks"
(arXiv 1706.04599): equal-width bins on the confidence, ECE = sum_b (n_b / n) |acc_b - conf_b|
and MCE = max_b |acc_b - conf_b| over the non-empty bins.

Further measures: calibration slope and intercept by logistic recalibration (Cox 1958,
Biometrika 45:562-565; Van Calster et al. 2019, "Calibration: the Achilles heel of predictive
analytics", BMC Medicine 17:230), class-wise ECE and equal-mass ("adaptive") ECE (Nixon et al.
2019, "Measuring Calibration in Deep Learning", CVPR Workshops, arXiv 1904.01685).
"""
import numpy as np

from src.analysis._common import check_pair, task_name

_BINS_ERROR = "bins must be >= 1"


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
        raise ValueError(_BINS_ERROR)
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


# --------------------------------------------------------------------------- slope / intercept

def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _nll(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.logaddexp(0.0, z) - y * z))


def _newton(x: np.ndarray, y: np.ndarray, slope_free: bool, max_iter: int = 50,
            tol: float = 1e-9):
    """Logistic regression of ``y`` on ``x`` by Newton-Raphson (= IRLS) with step halving.

    ``slope_free``: fit (a, b) in logit P(y=1) = a + b x. Otherwise only ``a`` with the slope
    fixed at 1 (an offset: calibration-in-the-large). Returns ``(params, covariance, ok)``;
    ``ok`` is False if it did not converge (e.g. perfect separation) or the problem is singular.
    """
    design = np.column_stack([np.ones_like(x), x]) if slope_free else np.ones((x.size, 1))
    offset = 0.0 if slope_free else x
    beta = np.array([0.0, 1.0]) if slope_free else np.zeros(1)
    z = design @ beta + offset
    nll = _nll(z, y)
    cov = None
    for _ in range(max_iter):
        mu = _sigmoid(z)
        w = mu * (1.0 - mu)
        hess = design.T @ (design * w[:, None])
        try:
            step = np.linalg.solve(hess, design.T @ (y - mu))
            cov = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            return beta, None, False
        scale = 1.0
        while scale > 1e-8:
            z_new = design @ (beta + scale * step) + offset
            nll_new = _nll(z_new, y)
            if nll_new <= nll + 1e-12:
                break
            scale /= 2.0
        else:
            return beta, None, False
        beta, z, nll = beta + scale * step, z_new, nll_new
        if np.max(np.abs(scale * step)) < tol:
            # Quasi-separation: coefficients explode and the information vanishes.
            ok = bool(np.all(np.isfinite(beta)) and np.max(np.abs(beta)) < 1e3)
            return beta, cov, ok
    return beta, cov, False


def _fit_calibration(prob: np.ndarray, label: np.ndarray, eps: float) -> dict:
    """slope / intercept of one binary problem (see ``calibration_curve_stats``)."""
    nan = float("nan")
    out = {"slope": nan, "intercept": nan, "intercept_citl": nan, "slope_se": nan,
           "intercept_se": nan, "n": int(prob.size), "converged": False}
    y = label.astype(np.float64)
    if prob.size == 0 or y.min() == y.max():
        return out
    x = _logit(prob, eps)
    beta, cov, ok = _newton(x, y, slope_free=True) if np.ptp(x) > 0 else (None, None, False)
    if ok:
        out.update(intercept=float(beta[0]), slope=float(beta[1]),
                   intercept_se=float(np.sqrt(cov[0, 0])), slope_se=float(np.sqrt(cov[1, 1])),
                   converged=True)
    a, _, ok_a = _newton(x, y, slope_free=False)
    if ok_a:
        out["intercept_citl"] = float(a[0])
    return out


def calibration_curve_stats(probs, targets, task, eps: float = 1e-6) -> dict:
    """Calibration slope and intercept (weak calibration, Van Calster et al. 2019).

    Logistic regression of the outcome on the logit of the predicted probability,
    ``logit P(y=1) = a + b * logit(p)`` (Cox 1958), fitted by Newton-Raphson / IRLS in NumPy. A
    perfectly calibrated model has ``b = 1`` and ``a = 0``; ``b < 1`` means overconfident
    predictions (too extreme), ``b > 1`` underconfident. Probabilities are clipped to
    ``[eps, 1 - eps]`` before the logit (float16 probabilities reach exactly 0 and 1).

    * Multiclass, **top-label**: ``p`` = max probability, ``y`` = the argmax is correct.
    * Multi-label / binary: every (sample, class) is one observation, ``p`` the probability of
      the class and ``y`` its label (all pooled).
    * ``per_class``: the same fit on each column (one-vs-rest in multiclass), as dicts.

    Each result is ``{"slope", "intercept", "intercept_citl", "slope_se", "intercept_se", "n",
    "converged"}``. ``intercept`` is the one of the joint fit (a, b); ``intercept_citl`` is the
    calibration-in-the-large with the slope fixed at 1 (the intercept Van Calster recommends
    reporting next to the slope); SEs are Wald. NaN and ``converged = False`` when the fit does
    not exist: a single outcome value, constant predictions or perfect separation.
    """
    name = task_name(task)
    p, y = check_pair(probs, targets)
    if name == "multiclass":
        top = _fit_calibration(p.max(axis=1), p.argmax(axis=1) == y.argmax(axis=1), eps)
    else:
        top = _fit_calibration(p.ravel(), y.ravel(), eps)
    return {**top, "per_class": [_fit_calibration(p[:, c], y[:, c], eps)
                                 for c in range(p.shape[1])]}


# --------------------------------------------------------------------------- classwise / adaptive

def _binned_gap(prob: np.ndarray, freq: np.ndarray, bin_idx: np.ndarray, bins: int):
    """(ece, mce, conf (bins,), freq (bins,), count (bins,)) for the bin of every element."""
    count = np.bincount(bin_idx, minlength=bins)
    sum_p = np.bincount(bin_idx, weights=prob, minlength=bins)
    sum_f = np.bincount(bin_idx, weights=freq, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        conf = np.where(count > 0, sum_p / count, np.nan)
        fr = np.where(count > 0, sum_f / count, np.nan)
    gap = np.abs(fr - conf)
    n = int(count.sum())
    ece = float(np.nansum(gap * count) / n) if n else float("nan")
    mce = float(np.nanmax(gap)) if n else float("nan")
    return ece, mce, conf, fr, count


def per_class_calibration(probs, targets, task, bins: int = 15) -> dict:
    """Class-wise calibration: ECE of the probability of each class (Nixon et al. 2019, SCE).

    For class k the probability ``p_k`` is binned in ``bins`` equal-width bins on [0, 1] (left
    closed, the last one also holds 1) and compared with the observed frequency of the class in
    each bin: ``ECE_k = sum_b (n_b / N) |freq_b - mean(p_k)_b|``. Unlike ``calibration`` (which
    scores the confidence of the *predicted* label), this is the reliability of the probability
    of every class, which is what a rare class in multi-label needs. Same rule in the three tasks
    (one-vs-rest in multiclass).

    Returns ``{"ece": (C,), "mce": (C,), "brier": (C,), "n_pos": (C,), "n": N, "sce" (mean of
    ece over the classes), "edges": (bins+1,), "confidence"/"frequency"/"count": (C, bins)}``
    (NaN in empty bins; everything NaN if there are no samples).
    """
    task_name(task)
    if bins < 1:
        raise ValueError(_BINS_ERROR)
    p, y = check_pair(probs, targets)
    n, n_cls = p.shape
    ece, mce, brier = np.full(n_cls, np.nan), np.full(n_cls, np.nan), np.full(n_cls, np.nan)
    conf, freq = np.full((n_cls, bins), np.nan), np.full((n_cls, bins), np.nan)
    count = np.zeros((n_cls, bins), dtype=np.int64)
    if n:
        for c in range(n_cls):
            idx = np.minimum((p[:, c] * bins).astype(np.int64), bins - 1)
            ece[c], mce[c], conf[c], freq[c], count[c] = _binned_gap(
                p[:, c], y[:, c].astype(np.float64), idx, bins)
            brier[c] = float(((p[:, c] - y[:, c]) ** 2).mean())
    ok = ~np.isnan(ece)
    return {"ece": ece, "mce": mce, "brier": brier, "n_pos": y.sum(axis=0).astype(np.int64),
            "n": n, "sce": float(ece[ok].mean()) if ok.any() else float("nan"),
            "edges": np.linspace(0.0, 1.0, bins + 1), "confidence": conf, "frequency": freq,
            "count": count}


def _equal_mass(prob: np.ndarray, freq: np.ndarray, bins: int):
    """Equal-mass binning: sort by ``prob`` and cut in ``bins`` runs of (almost) equal size.

    Returns ``(_binned_gap(...), order)``; ``order`` is the sort permutation."""
    order = np.argsort(prob, kind="mergesort")
    n_bins = min(bins, prob.size)
    idx = np.empty(prob.size, dtype=np.int64)
    for b, chunk in enumerate(np.array_split(np.arange(prob.size), n_bins)):
        idx[order[chunk]] = b
    return _binned_gap(prob, freq, idx, n_bins), order


def adaptive_ece(probs, targets, task, bins: int = 15, classwise: bool = False) -> dict:
    """Adaptive calibration error: ECE with **equal-mass** bins (Nixon et al. 2019).

    Equal-width bins (``calibration``) leave most bins empty and put almost all samples in the
    last ones when the network is confident; here the sorted predictions are cut in ``bins``
    groups of the same size (differing by at most one; with fewer predictions than bins there are
    as many bins as predictions), so every bin has the same weight in the estimate.

    * ``classwise=False`` (default): the predictions of ``calibration`` (top-label in multiclass,
      confidence of the predicted label per element otherwise), so it is directly comparable with
      its ECE.
    * ``classwise=True``: Nixon's ACE, the mean over the classes of the equal-mass error of the
      probability of each class against its frequency (``per_class`` holds the C values).

    Returns ``{"ece", "mce", "n", "bins": [{"lower", "upper", "confidence", "accuracy",
    "count"}, ...]}``; with ``classwise`` ``bins`` is empty, ``mce`` NaN and ``per_class`` given.
    """
    name = task_name(task)
    if bins < 1:
        raise ValueError(_BINS_ERROR)
    p, y = check_pair(probs, targets)
    nan = float("nan")
    if classwise:
        per = np.array([_equal_mass(p[:, c], y[:, c].astype(np.float64), bins)[0][0]
                        if p.shape[0] else nan for c in range(p.shape[1])])
        ok = ~np.isnan(per)
        return {"ece": float(per[ok].mean()) if ok.any() else nan, "mce": nan,
                "n": int(p.shape[0]), "bins": [], "per_class": per}
    if name == "multiclass":
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == y.argmax(axis=1)).astype(np.float64)
    else:
        pos = p >= 0.5
        conf = np.where(pos, p, 1.0 - p).ravel()
        correct = (pos == y).ravel().astype(np.float64)
    if conf.size == 0:
        return {"ece": nan, "mce": nan, "n": 0, "bins": []}
    (ece, mce, mean_conf, mean_acc, count), order = _equal_mass(conf, correct, bins)
    sorted_conf = conf[order]
    out_bins, start = [], 0
    for b in range(count.size):
        seg = sorted_conf[start:start + count[b]]
        out_bins.append({"lower": float(seg[0]), "upper": float(seg[-1]),
                         "confidence": float(mean_conf[b]), "accuracy": float(mean_acc[b]),
                         "count": int(count[b])})
        start += count[b]
    return {"ece": ece, "mce": mce, "n": int(conf.size), "bins": out_bins}
