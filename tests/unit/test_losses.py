"""Unit tests for the loss factory and the imbalance-aware losses."""
import pytest
import torch

from src.training.losses import (
    FocalLoss,
    build_criterion,
    pos_weight_from_counts,
)


class TestFocalLoss:
    def test_reduces_to_bce_when_gamma_zero_no_alpha(self):
        logits = torch.randn(8, 19)
        targets = torch.randint(0, 2, (8, 19)).float()
        focal = FocalLoss(gamma=0.0, alpha=-1.0)
        bce = torch.nn.BCEWithLogitsLoss()
        assert torch.allclose(focal(logits, targets), bce(logits, targets), atol=1e-6)

    def test_gamma_downweights_easy_examples(self):
        # Confident-correct predictions → focal loss < plain BCE
        logits = torch.full((4, 19), 6.0)      # very confident positive
        targets = torch.ones(4, 19)
        focal = FocalLoss(gamma=2.0)
        bce = torch.nn.BCEWithLogitsLoss()
        assert focal(logits, targets).item() < bce(logits, targets).item()

    def test_handles_soft_targets(self):
        # mixup produces targets in [0, 1] — must not crash and stay finite
        logits = torch.randn(4, 19)
        targets = torch.rand(4, 19)
        out = FocalLoss(gamma=2.0)(logits, targets)
        assert torch.isfinite(out)

    def test_reductions(self):
        logits = torch.randn(4, 19)
        targets = torch.randint(0, 2, (4, 19)).float()
        none = FocalLoss(reduction="none")(logits, targets)
        assert none.shape == (4, 19)
        assert FocalLoss(reduction="sum")(logits, targets).item() == pytest.approx(
            none.sum().item(), rel=1e-5
        )


class TestPosWeight:
    def test_neg_over_pos(self):
        # 100 samples, class 0 positive 10× → pos_weight = 90/10 = 9
        pos = torch.tensor([10.0, 50.0])
        w = pos_weight_from_counts(pos, n_samples=100)
        assert w[0].item() == pytest.approx(9.0)
        assert w[1].item() == pytest.approx(1.0)

    def test_never_seen_class_gets_clamp(self):
        pos = torch.tensor([0.0, 50.0])
        w = pos_weight_from_counts(pos, n_samples=100, clamp_max=100.0)
        assert w[0].item() == 100.0   # not +inf

    def test_clamp_caps_extreme_imbalance(self):
        pos = torch.tensor([1.0])      # 9999 neg / 1 pos = 9999 → clamped
        w = pos_weight_from_counts(pos, n_samples=10000, clamp_max=100.0)
        assert w[0].item() == 100.0


class TestBuildCriterion:
    def test_default_is_bce(self):
        crit = build_criterion({})
        assert isinstance(crit, torch.nn.BCEWithLogitsLoss)
        assert crit.pos_weight is None

    def test_bce_with_pos_weight(self):
        pw = torch.ones(19)
        crit = build_criterion({"loss": "bce"}, pos_weight=pw)
        assert isinstance(crit, torch.nn.BCEWithLogitsLoss)
        assert crit.pos_weight is pw

    def test_focal_selected(self):
        crit = build_criterion({"loss": "focal", "focal_gamma": 3.0})
        assert isinstance(crit, FocalLoss)
        assert crit.gamma == 3.0

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            build_criterion({"loss": "hinge"})


class TestBuilderWiring:
    """The builder must translate config → criterion (without building the model)."""

    def _builder(self, train_cfg):
        import torch
        from src.training.builder import TrainingSessionBuilder
        cfg = {
            "data": {"root": ".", "metadata": "x.parquet"},
            "model": {"name": "vit_tiny_patch16_224", "num_classes": 19},
            "training": {"epochs": 1, "batch_size": 4, "lr": 0.001, **train_cfg},
            "checkpoint": {"dir": "checkpoints"},
        }
        return TrainingSessionBuilder(cfg, torch.device("cpu")), cfg

    def test_default_bce_returns_none(self):
        b, cfg = self._builder({})
        assert b._build_criterion(cfg, "vit_tiny_patch16_224") is None

    def test_focal_wired(self):
        b, cfg = self._builder({"loss": "focal", "focal_gamma": 2.0})
        crit = b._build_criterion(cfg, "vit_tiny_patch16_224")
        assert isinstance(crit, FocalLoss)

    def test_pos_weight_list_wired(self):
        b, cfg = self._builder({"pos_weight": [1.0] * 19})
        crit = b._build_criterion(cfg, "vit_tiny_patch16_224")
        assert isinstance(crit, torch.nn.BCEWithLogitsLoss)
        assert crit.pos_weight is not None and crit.pos_weight.shape[0] == 19
