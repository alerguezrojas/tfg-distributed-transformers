"""Out-of-fold label cleaning for graded labels (pure pandas/NumPy, no torch, no I/O).

The recipe of the PANDA winners (``docs/research/panda.md``): split the training set into
K folds, train one model per fold on the other K-1, and predict the held-out fold. Each
training sample then has a prediction from a model that never saw it. A sample whose label
is far from that prediction is likely mislabelled and is left out of a new training run.

Everything works on a ``splits`` table (``image_id, split``) like PANDA's ``splits.csv``:
the folds are new tables in which the held-out fold is ``val``, the rest of the training
set is ``train`` and every other row becomes ``excluded`` (a split no reader loads). The
validation and test sets of the original partition are never touched by the cleaning.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

EXCLUDED = "excluded"


def _hash_key(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"oof:{seed}:{image_id}".encode()).hexdigest()


def assign_folds(ids: pd.Series, strata: pd.Series, n_folds: int, seed: int = 42) -> pd.Series:
    """Fold (0..n_folds-1) of each id, stratified: inside each stratum the ids are put in a
    seeded hash order and dealt round-robin, so every fold gets the same mix of strata.
    Deterministic and independent of the input order."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    folds = pd.Series(-1, index=ids.index, dtype=int)
    for _, group in ids.groupby(strata.astype(str), sort=True):
        order = sorted(group.index, key=lambda i: _hash_key(seed, str(group[i])))
        for rank, idx in enumerate(order):
            folds[idx] = rank % n_folds
    return folds


def fold_splits(splits: pd.DataFrame, labels: pd.DataFrame, n_folds: int = 5,
                seed: int = 42, strata: tuple[str, ...] = ("data_provider", "isup_grade"),
                ) -> list[pd.DataFrame]:
    """One ``image_id, split`` table per fold: held-out fold -> ``val``, the rest of
    ``train`` -> ``train``, the original ``val``/``test`` rows -> ``excluded``.

    ``labels`` gives the stratification columns (e.g. PANDA's ``train.csv``)."""
    train = splits[splits["split"] == "train"].merge(labels, on="image_id", how="left")
    if train[list(strata)].isna().any().any():
        raise ValueError("some training ids have no label/stratum in `labels`")
    key = train[list(strata)].astype(str).agg("|".join, axis=1)
    fold_of = dict(zip(train["image_id"], assign_folds(train["image_id"], key, n_folds, seed)))
    out = []
    for k in range(n_folds):
        def role(row, k=k):
            if row["split"] != "train":
                return EXCLUDED
            return "val" if fold_of[row["image_id"]] == k else "train"
        table = splits[["image_id"]].copy()
        table["split"] = splits.apply(role, axis=1)
        out.append(table)
    return out


def oof_scores(probs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Per-sample disagreement between an out-of-fold prediction and the label.

    ``probs`` (N, C) over ordered grades, ``labels`` (N,) ints. Columns: ``expected``
    (the predicted mean grade, sum_k k p_k), ``gap`` (|expected - label|), ``p_label``
    (probability given to the label) and ``pred`` (argmax)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2 or probs.shape[0] != labels.shape[0]:
        raise ValueError(f"probs {probs.shape} and labels {labels.shape} do not match")
    expected = probs @ np.arange(probs.shape[1])
    return pd.DataFrame({"expected": expected, "gap": np.abs(expected - labels),
                         "p_label": probs[np.arange(len(labels)), labels],
                         "pred": probs.argmax(axis=1)})


def cleaned_splits(splits: pd.DataFrame, flagged: set[str] | list[str]) -> pd.DataFrame:
    """The original partition with the flagged **training** ids moved to ``excluded``
    (a flagged id in ``val``/``test`` is an error: those sets are never cleaned)."""
    flagged = set(flagged)
    in_split = splits.set_index("image_id")["split"]
    unknown = flagged - set(in_split.index)
    if unknown:
        raise ValueError(f"{len(unknown)} flagged ids are not in the splits table")
    not_train = [i for i in flagged if in_split[i] != "train"]
    if not_train:
        raise ValueError(f"{len(not_train)} flagged ids are not training samples "
                         f"(e.g. {not_train[0]}): only the training set is cleaned")
    out = splits.copy()
    out.loc[out["image_id"].isin(flagged), "split"] = EXCLUDED
    return out
