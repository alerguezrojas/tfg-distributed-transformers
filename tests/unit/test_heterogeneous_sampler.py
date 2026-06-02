"""Unit tests for src/training/heterogeneous_sampler.py"""
import pytest
import torch
from torch.utils.data import TensorDataset
from src.training.heterogeneous_sampler import HeterogeneousDistributedSampler


@pytest.fixture
def dataset():
    return TensorDataset(torch.arange(100))


class TestHeterogeneousDistributedSampler:

    def test_proportional_split(self, dataset):
        s0 = HeterogeneousDistributedSampler(dataset, weights=[3, 1], rank=0, world_size=2)
        s1 = HeterogeneousDistributedSampler(dataset, weights=[3, 1], rank=1, world_size=2)
        assert len(s0) == 75
        assert len(s1) == 25

    def test_total_equals_dataset(self, dataset):
        s0 = HeterogeneousDistributedSampler(dataset, weights=[16, 1], rank=0, world_size=2)
        s1 = HeterogeneousDistributedSampler(dataset, weights=[16, 1], rank=1, world_size=2)
        assert len(s0) + len(s1) == len(dataset)

    def test_no_overlap(self, dataset):
        s0 = HeterogeneousDistributedSampler(dataset, weights=[16, 1], rank=0, world_size=2)
        s1 = HeterogeneousDistributedSampler(dataset, weights=[16, 1], rank=1, world_size=2)
        idx0 = set(list(s0))
        idx1 = set(list(s1))
        assert len(idx0 & idx1) == 0

    def test_covers_all_indices(self, dataset):
        s0 = HeterogeneousDistributedSampler(dataset, weights=[2, 3], rank=0, world_size=2)
        s1 = HeterogeneousDistributedSampler(dataset, weights=[2, 3], rank=1, world_size=2)
        all_idx = set(list(s0)) | set(list(s1))
        assert all_idx == set(range(len(dataset)))

    def test_equal_weights_equal_split(self, dataset):
        s0 = HeterogeneousDistributedSampler(dataset, weights=[1, 1], rank=0, world_size=2)
        s1 = HeterogeneousDistributedSampler(dataset, weights=[1, 1], rank=1, world_size=2)
        assert abs(len(s0) - len(s1)) <= 1

    def test_three_ranks(self, dataset):
        samplers = [
            HeterogeneousDistributedSampler(dataset, weights=[4, 2, 1], rank=i, world_size=3)
            for i in range(3)
        ]
        # Total = dataset size
        assert sum(len(s) for s in samplers) == len(dataset)
        # No overlap
        all_sets = [set(list(s)) for s in samplers]
        for i in range(3):
            for j in range(i + 1, 3):
                assert len(all_sets[i] & all_sets[j]) == 0

    def test_shuffle_changes_order(self, dataset):
        s0 = HeterogeneousDistributedSampler(dataset, weights=[1, 1], rank=0, world_size=2,
                                              shuffle=True, seed=0)
        s0.set_epoch(0)
        idx_epoch0 = list(s0)
        s0.set_epoch(1)
        idx_epoch1 = list(s0)
        assert idx_epoch0 != idx_epoch1

    def test_no_shuffle_deterministic(self, dataset):
        s0 = HeterogeneousDistributedSampler(dataset, weights=[1, 1], rank=0, world_size=2,
                                              shuffle=False)
        idx1 = list(s0)
        idx2 = list(s0)
        assert idx1 == idx2

    def test_drop_last(self):
        ds = TensorDataset(torch.arange(101))  # not evenly divisible
        s0 = HeterogeneousDistributedSampler(ds, weights=[3, 1], rank=0, world_size=2,
                                              drop_last=True)
        s1 = HeterogeneousDistributedSampler(ds, weights=[3, 1], rank=1, world_size=2,
                                              drop_last=True)
        assert len(s0) + len(s1) <= len(ds)

    def test_wrong_world_size_raises(self, dataset):
        with pytest.raises(ValueError):
            HeterogeneousDistributedSampler(dataset, weights=[1, 1, 1], rank=0, world_size=2)

    def test_rank_out_of_range_raises(self, dataset):
        with pytest.raises(ValueError, match="rank"):
            HeterogeneousDistributedSampler(dataset, weights=[1, 1], rank=2, world_size=2)
