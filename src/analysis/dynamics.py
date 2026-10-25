"""Training dynamics per sample: data cartography, flips between epochs and worst errors.

Data cartography: Swayamdipta et al. (2020), "Dataset Cartography: Mapping and Diagnosing
Datasets with Training Dynamics" (arXiv 2009.10795).
"""
import numpy as np

from src.analysis._common import check_pair, predict, task_name

#: Fraction of samples (the most variable ones) labelled ``ambiguous`` (the paper's top third).
AMBIGUOUS_FRAC = 1 / 3
#: Mean probability of the truth separating ``hard`` (below) from ``easy`` (at or above).
CONFIDENCE_SPLIT = 0.5


def prob_of_true(probs, targets, task) -> np.ndarray:
    """Probability the model gives to the truth, per sample -> (N,).

    Multiclass: the probability of the true class. Multi-label / binary: the mean over all the
    classes of ``p`` where the label is positive and ``1 - p`` where it is negative (so the
    many easy negatives of a multi-label problem pull it up; it still ranks samples).
    """
    name = task_name(task)
    p, y = check_pair(probs, targets)
    if name == "multiclass":
        return p[np.arange(len(p)), y.argmax(axis=1)]
    return np.where(y, p, 1.0 - p).mean(axis=1)


def sample_correct(probs, targets, task, threshold: float = 0.5) -> np.ndarray:
    """Whether each sample is right -> bool (N,).

    Multiclass: the argmax is the true class. Multi-label / binary: every label is right
    (exact match / subset accuracy, ``probs > threshold``).
    """
    name = task_name(task)
    p, y = check_pair(probs, targets)
    pred = predict(p, name, threshold)
    if name == "multiclass":
        return pred.argmax(axis=1) == y.argmax(axis=1)
    return (pred == y).all(axis=1)


def cartography(prob_true_by_epoch, correct_by_epoch,
                ambiguous_frac: float = AMBIGUOUS_FRAC,
                confidence_split: float = CONFIDENCE_SPLIT) -> dict:
    """Confidence, variability, correctness and region of every sample.

    Inputs are (E, M) arrays: the probability of the truth (``prob_of_true``) and whether the
    sample was right (``sample_correct``) at each of the E epochs. Per sample (paper's
    definitions): ``confidence`` = mean over epochs of the probability of the truth,
    ``variability`` = its standard deviation (population, ddof=0), ``correctness`` = fraction of
    epochs in which it was right.

    Regions, with the thresholds in the result (``thresholds``):

    * ``ambiguous``: variability strictly above the (1 - ``ambiguous_frac``) quantile of the
      variability (the most variable third, as in the paper's map; with one epoch, or all equal,
      nothing is ambiguous);
    * of the rest, ``easy`` if confidence >= ``confidence_split`` and ``hard`` otherwise
      (low variability: the model is consistently right, or consistently wrong).

    Returns ``{"confidence", "variability", "correctness", "region" (str array), "thresholds"}``.
    """
    pt = np.asarray(prob_true_by_epoch, dtype=np.float64)
    ok = np.asarray(correct_by_epoch, dtype=np.float64)
    if pt.ndim != 2 or ok.shape != pt.shape:
        raise ValueError("prob_true_by_epoch and correct_by_epoch must be (epochs, samples) "
                         f"and equal; got {pt.shape} and {ok.shape}")
    confidence = pt.mean(axis=0)
    variability = pt.std(axis=0)
    correctness = ok.mean(axis=0)
    var_thr = float(np.quantile(variability, 1.0 - ambiguous_frac)) if variability.size else 0.0
    region = np.where(variability > var_thr, "ambiguous",
                      np.where(confidence >= confidence_split, "easy", "hard"))
    return {
        "confidence": confidence,
        "variability": variability,
        "correctness": correctness,
        "region": region,
        "thresholds": {"variability": var_thr, "confidence": float(confidence_split)},
    }


def flips(probs_prev, probs_curr, targets, task, threshold: float = 0.5) -> dict:
    """Samples that change between two epochs -> ``{"to_wrong": idx, "to_right": idx}``.

    ``to_wrong``: right before, wrong now; ``to_right``: wrong before, right now (the criterion
    is ``sample_correct``). Indices are positions in the arrays given.
    """
    before = sample_correct(probs_prev, targets, task, threshold)
    after = sample_correct(probs_curr, targets, task, threshold)
    return {"to_wrong": np.nonzero(before & ~after)[0],
            "to_right": np.nonzero(~before & after)[0]}


def worst_errors(probs, targets, loss, task, k: int, threshold: float = 0.5) -> dict:
    """The ``k`` samples with the highest loss, from worst to best.

    Returns ``{"indices" (k,), "loss" (k,), "targets" (k, C) uint8, "preds" (k, C) uint8}``;
    ``preds`` is the hard prediction (argmax one-hot in multiclass, ``probs > threshold``
    otherwise). Fewer than ``k`` rows if there are fewer samples; NaN losses go last.
    """
    name = task_name(task)
    p, y = check_pair(probs, targets)
    loss = np.asarray(loss, dtype=np.float64).ravel()
    if loss.shape[0] != p.shape[0]:
        raise ValueError(f"loss has {loss.shape[0]} entries for {p.shape[0]} samples")
    order = np.argsort(-np.nan_to_num(loss, nan=-np.inf), kind="stable")[:max(int(k), 0)]
    return {
        "indices": order,
        "loss": loss[order],
        "targets": y[order].astype(np.uint8),
        "preds": predict(p[order], name, threshold).astype(np.uint8),
    }
