"""Ordinal loss ``ce_emd``: cross-entropy + squared Earth Mover's Distance for graded classes."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import registry
from src.data.registry import get_dataset_spec, register_dataset
from src.data.spec import DatasetSpec
from src.training.config_validator import ConfigError, validate_config
from src.training.losses import CrossEntropyEMDLoss, build_criterion, squared_emd
from src.training.tasks import MultiClassTask, get_task

_BIG = 50.0   # logit margin that makes the softmax practically one-hot


def _confident(classes, c=6):
    """Logits whose softmax is (almost) one-hot on ``classes``."""
    return F.one_hot(torch.tensor(classes), c).float() * _BIG


def _onehot(classes, c=6):
    return F.one_hot(torch.tensor(classes), c).float()


class TestSquaredEMD:
    def test_zero_when_prediction_matches(self):
        emd = squared_emd(_confident([0, 3, 5]), _onehot([0, 3, 5]))
        assert torch.allclose(emd, torch.zeros(3), atol=1e-6)

    def test_grows_with_the_distance_between_grades(self):
        # Moving all the mass by d grades leaves d CDF steps of size 1 -> EMD^2 = d.
        emd = squared_emd(_confident([1, 2, 5]), _onehot([0, 0, 0]))
        assert emd.tolist() == pytest.approx([1.0, 2.0, 5.0], abs=1e-5)

    def test_soft_targets(self):
        # Target half grade 0, half grade 2; a confident grade-1 prediction is 0.5 off
        # at the first two CDF steps: 0.5^2 + 0.5^2.
        target = torch.tensor([[0.5, 0.0, 0.5, 0.0]])
        emd = squared_emd(_confident([1], c=4), target)
        assert emd.item() == pytest.approx(0.5, abs=1e-5)

    def test_float32_under_half_precision_logits(self):
        emd = squared_emd(_confident([2]).half(), _onehot([0]))
        assert emd.dtype == torch.float32


class TestCrossEntropyEMDLoss:
    def test_weight_zero_is_plain_cross_entropy(self):
        logits, target = torch.randn(8, 6), _onehot([0, 1, 2, 3, 4, 5, 2, 1])
        assert CrossEntropyEMDLoss(0.0)(logits, target).item() == pytest.approx(
            nn.CrossEntropyLoss()(logits, target).item())

    def test_adds_weighted_emd(self):
        logits, target = torch.randn(8, 6), _onehot([0, 1, 2, 3, 4, 5, 2, 1])
        expected = (nn.CrossEntropyLoss()(logits, target)
                    + 2.0 * squared_emd(logits, target).mean())
        assert CrossEntropyEMDLoss(2.0)(logits, target).item() == pytest.approx(expected.item())

    def test_far_error_costs_more_than_near_error(self):
        # Same (wrong) confidence, different distance: the plain cross-entropy is equal,
        # the ordinal loss is not.
        target = _onehot([0])
        near, far = torch.tensor([[0.0, 3.0, 0, 0, 0, 0]]), torch.tensor([[0.0, 0, 0, 0, 0, 3.0]])
        ce = nn.CrossEntropyLoss()
        assert ce(near, target).item() == pytest.approx(ce(far, target).item())
        loss = CrossEntropyEMDLoss()
        assert loss(far, target).item() > loss(near, target).item()

    def test_gradient_is_finite(self):
        logits = torch.randn(4, 6, requires_grad=True)
        CrossEntropyEMDLoss()(logits, _onehot([0, 2, 4, 5])).backward()
        assert torch.isfinite(logits.grad).all()

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_bad_weight_rejected(self, bad):
        with pytest.raises(ValueError, match="emd_weight"):
            CrossEntropyEMDLoss(bad)


class TestTaskAndFactory:
    def test_ordinal_task_builds_it(self):
        crit = MultiClassTask(ordinal=True).build_criterion("ce_emd", emd_weight=0.5)
        assert isinstance(crit, CrossEntropyEMDLoss) and crit.emd_weight == 0.5

    def test_unordered_classes_reject_it(self):
        with pytest.raises(ValueError, match="ordered"):
            MultiClassTask().build_criterion("ce_emd")

    def test_multilabel_rejects_it(self):
        with pytest.raises(ValueError, match="not valid"):
            get_task("multilabel").build_criterion("ce_emd")

    def test_default_stays_cross_entropy(self):
        assert isinstance(MultiClassTask(ordinal=True).build_criterion(), nn.CrossEntropyLoss)

    def test_factory_takes_the_task_instance(self):
        crit = build_criterion({"loss": "ce_emd", "emd_weight": 2.0},
                               task=get_task("multiclass", ordinal=True))
        assert isinstance(crit, CrossEntropyEMDLoss) and crit.emd_weight == 2.0
        with pytest.raises(ValueError, match="ordered"):   # a bare name has no order
            build_criterion({"loss": "ce_emd"}, task="multiclass")

    def test_builder_passes_the_spec_task(self):
        from src.training.builder import TrainingSessionBuilder
        cfg = {
            "data": {"name": "panda", "root": "."},
            "model": {"name": "vit_tiny_patch16_224"},
            "training": {"epochs": 2, "batch_size": 4, "lr": 0.001, "loss": "ce_emd"},
            "checkpoint": {"dir": "checkpoints"},
        }
        crit = TrainingSessionBuilder(cfg, torch.device("cpu"))._build_criterion(
            cfg, get_dataset_spec("panda"))
        assert isinstance(crit, CrossEntropyEMDLoss)


def _cfg(name, training):
    return {
        "data": {"name": name},
        "model": {"name": "vit_tiny_patch16_224"},
        "training": {"epochs": 2, "batch_size": 4, "lr": 0.001, **training},
    }


@pytest.fixture
def graded_dataset():
    spec = register_dataset(DatasetSpec(
        name="graded_loss_test", description="toy", classes=("g0", "g1", "g2"),
        task="multiclass", in_channels=3, required_keys=(),
        build=lambda cfg, split, tf: None, transforms=lambda split: None, ordinal=True))
    yield spec
    registry._REGISTRY.pop(spec.name, None)


@pytest.fixture
def unordered_dataset():
    spec = register_dataset(DatasetSpec(
        name="unordered_loss_test", description="toy", classes=("a", "b", "c"),
        task="multiclass", in_channels=3, required_keys=(),
        build=lambda cfg, split, tf: None, transforms=lambda split: None))
    yield spec
    registry._REGISTRY.pop(spec.name, None)


class TestValidator:
    def test_accepts_ce_emd_on_graded_classes(self, graded_dataset):
        validate_config(_cfg(graded_dataset.name, {"loss": "ce_emd", "emd_weight": 0.5}))

    def test_rejects_ce_emd_without_order(self, unordered_dataset):
        with pytest.raises(ConfigError, match="ordered"):
            validate_config(_cfg(unordered_dataset.name, {"loss": "ce_emd"}))

    @pytest.mark.parametrize("bad", [-1.0, "1", True, float("nan"), float("inf")])
    def test_rejects_bad_weight(self, graded_dataset, bad):
        with pytest.raises(ConfigError, match="emd_weight"):
            validate_config(_cfg(graded_dataset.name, {"loss": "ce_emd", "emd_weight": bad}))

    def test_weight_without_ce_emd_is_an_error(self, graded_dataset):
        with pytest.raises(ConfigError, match="only applies"):
            validate_config(_cfg(graded_dataset.name, {"emd_weight": 1.0}))
