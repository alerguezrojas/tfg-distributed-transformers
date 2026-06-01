"""HeterogeneousDistributedSampler — assigns proportional dataset fractions per rank.

Standard DistributedSampler gives every rank the same number of samples (1/N of the
dataset).  In heterogeneous hardware (e.g. one V100 GPU + several CPU nodes) that is
wasteful: the fast device waits for the slow ones.

This sampler distributes indices so that each rank receives a fraction proportional to
its declared compute_weight.  A rank with weight=16 receives 16× more samples per step
than one with weight=1.

Usage
-----
    weights = [16, 1]   # rank 0 → GPU (fast), rank 1 → CPU (slow)
    sampler = HeterogeneousDistributedSampler(
        dataset, weights=weights, rank=rank, world_size=world_size,
        shuffle=True, drop_last=True,
    )

The batch_size for each rank is specified in the DataLoader independently.
The sampler only controls how many indices each rank owns; the loader slices
those indices further into mini-batches of the configured size.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class HeterogeneousDistributedSampler(Sampler):
    """Distributed sampler with per-rank proportional data allocation.

    Parameters
    ----------
    dataset:        The dataset to sample from.
    weights:        List of relative compute weights, one per rank.
                    e.g. [16, 1] means rank 0 gets 16/(16+1) of the data.
    rank:           Rank of the current process.
    world_size:     Total number of processes.
    shuffle:        Shuffle indices before splitting (use True for training).
    drop_last:      Drop the tail of the dataset to make it evenly divisible
                    across the weighted allocation.
    seed:           Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        weights: list[int | float],
        rank: int,
        world_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        if world_size != len(weights):
            raise ValueError(
                f"len(weights)={len(weights)} must equal world_size={world_size}"
            )
        if rank >= world_size:
            raise ValueError(f"rank={rank} must be < world_size={world_size}")

        self.dataset = dataset
        self.weights = weights
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        total_w = sum(weights)
        n = len(dataset)

        # Compute how many samples each rank gets (proportional to weight)
        raw_counts = [w / total_w * n for w in weights]
        # Floor all, then distribute remainder to heaviest ranks
        counts = [math.floor(c) for c in raw_counts]
        remainder = n - sum(counts) if not drop_last else 0
        fracs = [(rc - c, i) for i, (rc, c) in enumerate(zip(raw_counts, counts))]
        fracs.sort(reverse=True)
        for j in range(remainder):
            counts[fracs[j][1]] += 1

        # Compute start offset for this rank
        self.num_samples = counts[rank]
        self.total_size = sum(counts)
        self._start = sum(counts[:rank])

    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            # Pad to total_size
            padding = self.total_size - len(indices)
            indices = indices + indices[:padding]

        # Each rank gets its own slice
        my_indices = indices[self._start: self._start + self.num_samples]
        return iter(my_indices)
