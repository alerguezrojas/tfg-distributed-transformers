"""Tests for seeding (reproducibility) and the shared threshold grid."""
import random

import numpy as np
import torch

from src.training.reproducibility import set_seed, make_generator, seed_worker
from src.training import metrics as m


class TestSetSeed:
    def test_torch_reproducible(self):
        set_seed(42)
        a = torch.rand(5)
        set_seed(42)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_numpy_and_random_reproducible(self):
        set_seed(7)
        a_np, a_py = np.random.rand(3).tolist(), [random.random() for _ in range(3)]
        set_seed(7)
        b_np, b_py = np.random.rand(3).tolist(), [random.random() for _ in range(3)]
        assert a_np == b_np
        assert a_py == b_py

    def test_different_seeds_differ(self):
        set_seed(1)
        a = torch.rand(5)
        set_seed(2)
        b = torch.rand(5)
        assert not torch.equal(a, b)

    def test_seeded_model_init_is_deterministic(self):
        set_seed(123)
        w1 = torch.nn.Linear(768, 19).weight.detach().clone()
        set_seed(123)
        w2 = torch.nn.Linear(768, 19).weight.detach().clone()
        assert torch.equal(w1, w2)

    def test_make_generator_is_deterministic(self):
        g1, g2 = make_generator(42), make_generator(42)
        a = torch.randperm(10, generator=g1)
        b = torch.randperm(10, generator=g2)
        assert torch.equal(a, b)

    def test_seed_worker_runs(self):
        seed_worker(0)   # must not raise


class TestThresholdGrid:
    def test_grid_floor_below_focal_optimum(self):
        # Focal lowers probabilities → its optimum can fall below 0.30; the grid
        # must reach well under 0.5 so the search can find it.
        assert min(m.THRESHOLD_GRID) <= 0.15
        assert 0.5 in m.THRESHOLD_GRID
        assert max(m.THRESHOLD_GRID) >= 0.60

    def test_trainer_and_eval_share_the_grid(self):
        import src.training.trainer as trainer_mod
        import scripts.eval as eval_mod
        assert trainer_mod._THRESHOLD_GRID is m.THRESHOLD_GRID
        assert eval_mod._THRESHOLD_GRID is m.THRESHOLD_GRID

    def test_grid_sorted_and_in_range(self):
        assert m.THRESHOLD_GRID == sorted(m.THRESHOLD_GRID)
        assert all(0.0 < t < 1.0 for t in m.THRESHOLD_GRID)
