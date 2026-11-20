"""Shared helpers: input normalisation and the task-dependent hard predictions."""
import numpy as np

TASK_NAMES = ("multilabel", "binary", "multiclass")


def task_name(task) -> str:
    """Task as a string; accepts a ``Task`` object (anything with ``.name``) too."""
    name = getattr(task, "name", task)
    if name not in TASK_NAMES:
        raise ValueError(f"unknown task {name!r}; available: {list(TASK_NAMES)}")
    return name


def as_matrix(x, name: str = "array") -> np.ndarray:
    """float64 (N, C); a 1-D vector is taken as a single column. float16 inputs (as stored in
    the inspection files) are upcast so sums and comparisons do not lose precision."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (samples, classes); got shape {a.shape}")
    return a


def check_pair(probs, targets) -> tuple[np.ndarray, np.ndarray]:
    """(probs float64, hard targets bool), both (N, C) and aligned."""
    p = as_matrix(probs, "probs")
    y = as_matrix(targets, "targets") > 0.5
    if p.shape != y.shape:
        raise ValueError(f"probs {p.shape} and targets {y.shape} must have the same shape")
    return p, y


def predict(probs: np.ndarray, task: str, threshold: float = 0.5) -> np.ndarray:
    """Hard 0/1 predictions (bool, N × C) with the semantics of ``Task.predict``:
    ``probs > threshold`` per label, or the one-hot argmax in multiclass."""
    if task == "multiclass":
        out = np.zeros(probs.shape, dtype=bool)
        out[np.arange(len(probs)), probs.argmax(axis=1)] = True
        return out
    return probs > threshold
