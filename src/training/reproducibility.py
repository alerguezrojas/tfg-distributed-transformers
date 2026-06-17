"""Reproducibility helpers — make a run deterministic given a seed.

Without seeding, two otherwise-identical runs differ in head init, shuffling,
augmentations, mixup λ and dropout masks — so a focal-vs-BCE comparison would
measure noise on top of the loss change. Seeding both arms with the same value
isolates the only intended difference (the loss).
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) RNGs for a reproducible run.

    Call once, before building the model and the DataLoaders.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """DataLoader ``worker_init_fn``: reseed NumPy and Python ``random`` per worker.

    torch already seeds each worker's torch RNG from the base seed; this also
    covers NumPy and Python's ``random`` (used by the augmentation transforms,
    e.g. the discrete rotation), which run inside the worker processes.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A torch.Generator seeded for deterministic DataLoader shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
