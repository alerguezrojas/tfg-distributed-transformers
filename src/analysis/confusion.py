"""Confusion matrix with the indices of the samples in every cell."""
import numpy as np

from src.analysis._common import check_pair, predict, task_name


def confusion(probs, targets, task, threshold: float = 0.5) -> dict:
    """Counts and sample indices of the confusion matrix.

    * Multiclass: ``matrix[i, j]`` = samples whose true class (argmax of the one-hot target) is
      i and predicted class (argmax of the probabilities) is j; ``threshold`` is unused.
    * Multi-label / binary: the convention of ``ConfusionMatrixDecorator`` but in counts, not
      normalised: ``matrix[i, j]`` = samples with label i **and** predicted j (a co-activation
      matrix, so a sample counts in several cells; the diagonal is the TP of each class).
      Prediction = ``probs > threshold``.

    Returns::

        {"matrix": (C, C) int64,
         "cells": {(i, j): int64 array of sample indices} for every (i, j) (possibly empty),
         "per_class": [{"tp","fp","fn","tn": counts, "tp_idx","fp_idx","fn_idx","tn_idx"}, ...]}

    ``per_class`` is one-vs-rest in both tasks and is what is most useful in multi-label.
    """
    name = task_name(task)
    p, y = check_pair(probs, targets)
    pred = predict(p, name, threshold)
    n_cls = p.shape[1]
    if name == "multiclass":
        true_c, pred_c = y.argmax(axis=1), pred.argmax(axis=1)
        matrix = np.zeros((n_cls, n_cls), dtype=np.int64)
        np.add.at(matrix, (true_c, pred_c), 1)
        cells = {(i, j): np.nonzero((true_c == i) & (pred_c == j))[0]
                 for i in range(n_cls) for j in range(n_cls)}
    else:
        matrix = y.T.astype(np.int64) @ pred.astype(np.int64)
        cells = {(i, j): np.nonzero(y[:, i] & pred[:, j])[0]
                 for i in range(n_cls) for j in range(n_cls)}
    per_class = []
    for c in range(n_cls):
        tp, fp = y[:, c] & pred[:, c], ~y[:, c] & pred[:, c]
        fn, tn = y[:, c] & ~pred[:, c], ~y[:, c] & ~pred[:, c]
        per_class.append({
            "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()), "tn": int(tn.sum()),
            "tp_idx": np.nonzero(tp)[0], "fp_idx": np.nonzero(fp)[0],
            "fn_idx": np.nonzero(fn)[0], "tn_idx": np.nonzero(tn)[0],
        })
    return {"matrix": matrix, "cells": cells, "per_class": per_class}
