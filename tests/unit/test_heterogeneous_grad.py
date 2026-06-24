"""Heterogeneous DDP gradient normalization — scale correctness.

The HeterogeneousDDPTrainer scales the summed BCE so that, AFTER DDP averages the
per-rank gradients (÷ world_size), the result equals the standard single-process
BCE-mean gradient of the global concatenated batch. The fix:

    loss = loss_sum * world_size / (global_bs * n_classes)

These tests pin that scale without needing a real distributed run."""
import torch
import torch.nn as nn


def _hetero_loss(logits, labels, *, world_size: int, global_bs: int) -> torch.Tensor:
    """The exact normalization used in HeterogeneousDDPTrainer.train_epoch."""
    loss_sum = nn.BCEWithLogitsLoss(reduction="sum")(logits, labels)
    n_classes = logits.shape[1]
    return loss_sum * world_size / (global_bs * n_classes)


def test_single_rank_recovers_bce_mean():
    """world_size=1, global_bs=batch → must equal BCEWithLogitsLoss(reduction='mean')."""
    torch.manual_seed(0)
    logits = torch.randn(8, 19)
    labels = (torch.rand(8, 19) > 0.5).float()
    got = _hetero_loss(logits, labels, world_size=1, global_bs=8)
    want = nn.BCEWithLogitsLoss(reduction="mean")(logits, labels)
    assert torch.allclose(got, want, atol=1e-6)


def test_balanced_two_ranks_each_is_local_mean():
    """world_size=2, global_bs=2×local → each rank's loss is its LOCAL BCE-mean, so
    DDP averaging the two gradients yields the global mean gradient."""
    torch.manual_seed(1)
    logits = torch.randn(4, 19)          # one rank's local batch
    labels = (torch.rand(4, 19) > 0.5).float()
    got = _hetero_loss(logits, labels, world_size=2, global_bs=8)   # global = 2×4
    local_mean = nn.BCEWithLogitsLoss(reduction="mean")(logits, labels)
    assert torch.allclose(got, local_mean, atol=1e-6)


def test_old_formula_was_off_by_n_classes():
    """The previous loss_sum/global_bs over-scaled by n_classes/world_size."""
    torch.manual_seed(2)
    logits = torch.randn(8, 19)
    labels = (torch.rand(8, 19) > 0.5).float()
    loss_sum = nn.BCEWithLogitsLoss(reduction="sum")(logits, labels)
    old = loss_sum / 8                                   # old: loss_sum / global_bs (ws=1)
    mean = nn.BCEWithLogitsLoss(reduction="mean")(logits, labels)
    assert torch.allclose(old, mean * 19, atol=1e-4)     # off by exactly n_classes
