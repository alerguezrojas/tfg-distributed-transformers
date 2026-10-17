"""DatasetSpec — everything the framework needs to know about one dataset.

Each supported dataset is described once by a :class:`DatasetSpec` and registered
in :mod:`src.data.registry`. The trainers, the builder, the evaluation script and
the benchmark ask the spec (class names, input channels, task type, how to read a
split) instead of assuming BigEarthNet, so adding a dataset means writing a spec
and a reader — not touching the training code (open/closed principle).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Dataset

from src.training.tasks import TASKS, Task, get_task

#: (data_cfg, split, transform) → torch Dataset yielding (image, float target (C,)).
DatasetFactory = Callable[[dict, str, Callable | None], Dataset]
#: split → transform for that split ("train" augments, the others evaluate).
TransformFactory = Callable[[str], Callable]
#: (data_cfg, split) → (positives per class (C,), number of samples).
ClassCounter = Callable[[dict, str], "tuple[torch.Tensor, int]"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    classes: tuple[str, ...]
    task: str                       # 'multilabel' | 'binary' | 'multiclass'
    in_channels: int
    required_keys: tuple[str, ...]  # data.* keys the config must provide
    build: DatasetFactory
    transforms: TransformFactory
    class_counts: ClassCounter | None = None   # enables training.pos_weight: auto

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"{self.name}: unknown task {self.task!r}; "
                             f"available: {sorted(TASKS)}")
        if self.task == "binary" and len(self.classes) != 1:
            raise ValueError(f"{self.name}: a binary task has exactly one (positive) "
                             f"class, got {len(self.classes)}")
        if self.task == "multiclass" and len(self.classes) < 2:
            raise ValueError(f"{self.name}: a multiclass task needs at least 2 classes")
        if self.in_channels < 1:
            raise ValueError(f"{self.name}: in_channels must be >= 1")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def make_task(self) -> Task:
        return get_task(self.task)

    def dataset(self, data_cfg: dict, split: str) -> Dataset:
        """The ``split`` ('train' | 'val' | 'test') with its default transforms."""
        return self.build(data_cfg, split, self.transforms(split))
