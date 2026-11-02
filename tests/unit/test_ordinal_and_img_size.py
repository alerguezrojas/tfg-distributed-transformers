"""Ordered (graded) classes — the quadratic weighted kappa — and ``model.img_size``."""
import math

import pytest
import torch
import torch.nn.functional as F

from src.data import registry
from src.data.registry import register_dataset
from src.data.spec import DatasetSpec
from src.training import metrics as m
from src.training.config_validator import ConfigError, validate_config
from src.training.tasks import MultiClassTask, get_task


def _onehot(idx, c=6):
    return F.one_hot(torch.tensor(idx, dtype=torch.long), c).long()


def _qwk_reference(y_true, y_pred, c):
    """Textbook QWK with plain Python loops (independent of the vectorized one)."""
    n = len(y_true)
    obs = [[0.0] * c for _ in range(c)]
    for t, p in zip(y_true, y_pred):
        obs[t][p] += 1
    rows = [sum(r) for r in obs]
    cols = [sum(obs[i][j] for i in range(c)) for j in range(c)]
    num = den = 0.0
    for i in range(c):
        for j in range(c):
            w = (i - j) ** 2 / (c - 1) ** 2
            num += w * obs[i][j]
            den += w * rows[i] * cols[j] / n
    return 1.0 - num / den


class TestQuadraticWeightedKappa:
    def test_perfect_agreement_is_one(self):
        y = [0, 1, 2, 3, 4, 5, 2]
        assert m.quadratic_weighted_kappa(_onehot(y), _onehot(y)) == pytest.approx(1.0)

    def test_matches_reference(self):
        g = torch.Generator().manual_seed(0)
        y_true = torch.randint(0, 6, (200,), generator=g).tolist()
        y_pred = [min(5, max(0, t + int(d))) for t, d in
                  zip(y_true, torch.randint(-2, 3, (200,), generator=g))]
        got = m.quadratic_weighted_kappa(_onehot(y_pred), _onehot(y_true))
        assert got == pytest.approx(_qwk_reference(y_true, y_pred, 6), abs=1e-12)

    def test_far_errors_cost_more_than_near_ones(self):
        y_true = [0, 1, 2, 3, 4, 5] * 5
        near = [min(5, t + 1) for t in y_true]
        far = [5 if t < 3 else 0 for t in y_true]
        assert (m.quadratic_weighted_kappa(_onehot(near), _onehot(y_true))
                > m.quadratic_weighted_kappa(_onehot(far), _onehot(y_true)))

    def test_undefined_is_nan(self):
        same = _onehot([2, 2, 2])          # both raters constant on one class
        assert math.isnan(m.quadratic_weighted_kappa(same, same))
        assert math.isnan(m.quadratic_weighted_kappa(_onehot([], 6), _onehot([], 6)))


class TestOrdinalTask:
    def test_plain_multiclass_has_no_qwk(self):
        y = _onehot([0, 1, 2])
        assert "qwk" not in get_task("multiclass").metrics(y, y)

    def test_ordinal_adds_qwk_and_keeps_the_name(self):
        task = get_task("multiclass", ordinal=True)
        assert isinstance(task, MultiClassTask) and task.name == "multiclass"
        y = _onehot([0, 1, 2, 3])
        assert task.metrics(y, y)["qwk"] == pytest.approx(1.0)

    def test_ordinal_only_for_multiclass(self):
        with pytest.raises(ValueError, match="ordinal"):
            get_task("multilabel", ordinal=True)


def _spec(**overrides) -> DatasetSpec:
    fields = {
        "name": "graded_toy", "description": "toy", "classes": ("g0", "g1", "g2"),
        "task": "multiclass", "in_channels": 3, "required_keys": (),
        "build": lambda cfg, split, tf: None, "transforms": lambda split: None,
    }
    return DatasetSpec(**{**fields, **overrides})


def _cfg(name, training=None, model=None):
    return {
        "data": {"name": name},
        "model": {"name": "vit_tiny_patch16_224", **(model or {})},
        "training": {"epochs": 2, "batch_size": 4, "lr": 0.001, **(training or {})},
    }


@pytest.fixture
def graded_dataset():
    spec = register_dataset(_spec(name="graded_registry_test", ordinal=True))
    yield spec
    registry._REGISTRY.pop(spec.name, None)


class TestOrdinalSpec:
    def test_spec_task_is_ordinal(self):
        assert _spec(ordinal=True).make_task().ordinal is True
        assert _spec().make_task().ordinal is False

    def test_ordinal_needs_multiclass(self):
        with pytest.raises(ValueError, match="ordinal"):
            _spec(task="multilabel", ordinal=True)

    def test_select_by_qwk(self, graded_dataset):
        validate_config(_cfg(graded_dataset.name, training={"select_by": "qwk"}))
        with pytest.raises(ConfigError, match="qwk"):
            validate_config(_cfg("bigearthnet", training={"select_by": "qwk"}))


class TestImgSize:
    def test_default_side_is_224(self):
        from src.models.vit import build_model, input_size
        model = build_model("vit_tiny_patch16_224", num_classes=2, pretrained=False)
        assert input_size(model) == 224

    def test_vit_built_for_another_side(self):
        from src.models.vit import build_model, input_size
        model = build_model("vit_tiny_patch16_224", num_classes=3, pretrained=False,
                            img_size=64)
        assert input_size(model) == 64
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 3)

    def test_model_parallel_wrapper_keeps_it(self):
        from src.models.model_parallel import build_model_parallel_vit
        from src.models.vit import input_size
        model = build_model_parallel_vit("vit_tiny_patch16_224", num_classes=2,
                                         pretrained=False, devices=["cpu", "cpu"],
                                         img_size=64)
        assert input_size(model) == 64

    @pytest.mark.parametrize("bad", [0, -32, 224.0, "512", True])
    def test_validator_rejects_bad_values(self, bad):
        with pytest.raises(ConfigError, match="img_size"):
            validate_config(_cfg("bigearthnet", model={"img_size": bad}) |
                            {"data": {"root": "/r", "metadata": "/m.parquet"}})

    def test_validator_accepts_positive_int(self):
        validate_config(_cfg("bigearthnet", model={"img_size": 512}) |
                        {"data": {"root": "/r", "metadata": "/m.parquet"}})
