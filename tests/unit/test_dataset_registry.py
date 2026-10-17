"""Unit tests for DatasetSpec, the dataset registry and the config validation
that depends on them."""
import pytest
import torch

from src.data import registry
from src.data.registry import (
    DEFAULT_DATASET, available_datasets, dataset_spec_from_config,
    get_dataset_spec, register_dataset,
)
from src.data.spec import DatasetSpec
from src.training.config_validator import ConfigError, validate_config


def _spec(**overrides) -> DatasetSpec:
    fields = {
        "name": "toy", "description": "toy", "classes": ("a", "b"),
        "task": "multiclass", "in_channels": 1, "required_keys": (),
        "build": lambda cfg, split, tf: None, "transforms": lambda split: None,
    }
    return DatasetSpec(**{**fields, **overrides})


@pytest.fixture
def toy_dataset():
    spec = register_dataset(_spec(name="toy_registry_test"))
    yield spec
    registry._REGISTRY.pop(spec.name, None)


class TestBigEarthNetSpec:
    def test_is_default_and_registered(self):
        assert DEFAULT_DATASET == "bigearthnet"
        assert "bigearthnet" in available_datasets()
        assert dataset_spec_from_config({"data": {}}) is get_dataset_spec("bigearthnet")

    def test_properties(self):
        spec = get_dataset_spec("bigearthnet")
        assert spec.num_classes == 19
        assert spec.task == "multilabel"
        assert spec.in_channels == 3
        assert spec.required_keys == ("root", "metadata")
        assert spec.class_counts is not None

    def test_web_and_dataset_share_class_names(self):
        from src.web.dataset_stats import CLASS_NAMES
        assert tuple(CLASS_NAMES) == get_dataset_spec("bigearthnet").classes


class TestRegistry:
    def test_unknown_dataset(self):
        with pytest.raises(ValueError, match="unknown dataset"):
            get_dataset_spec("nope")

    def test_name_from_config_is_case_insensitive(self, toy_dataset):
        assert dataset_spec_from_config({"data": {"name": "TOY_REGISTRY_TEST"}}) is toy_dataset

    def test_duplicate_name_rejected(self, toy_dataset):
        clash = _spec(name=toy_dataset.name)
        with pytest.raises(ValueError, match="already registered"):
            register_dataset(clash)

    def test_reregistering_same_object_is_idempotent(self, toy_dataset):
        assert register_dataset(toy_dataset) is toy_dataset


class TestSpecValidation:
    def test_binary_needs_exactly_one_class(self):
        with pytest.raises(ValueError, match="exactly one"):
            _spec(task="binary", classes=("pos", "neg"))

    def test_multiclass_needs_two_classes(self):
        with pytest.raises(ValueError, match="at least 2"):
            _spec(classes=("only",))

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="unknown task"):
            _spec(task="regression")

    def test_channels_positive(self):
        with pytest.raises(ValueError, match="in_channels"):
            _spec(in_channels=0)

    def test_dataset_uses_split_transforms(self):
        seen = {}
        spec = _spec(build=lambda cfg, split, tf: seen.update(split=split, tf=tf),
                     transforms=lambda split: f"tf-{split}")
        spec.dataset({}, "val")
        assert seen == {"split": "val", "tf": "tf-val"}


def _cfg(data=None, model=None, training=None):
    return {
        "data": {"root": "/r", "metadata": "/m.parquet", **(data or {})},
        "model": {"name": "vit_tiny_patch16_224", **(model or {})},
        "training": {"epochs": 2, "batch_size": 4, "lr": 0.001, **(training or {})},
    }


class TestValidatorAgainstSpec:
    def test_num_classes_is_optional(self):
        validate_config(_cfg())

    def test_num_classes_must_match_dataset(self):
        cfg = _cfg(model={"num_classes": 10})
        with pytest.raises(ConfigError, match="does not match dataset"):
            validate_config(cfg)

    def test_unknown_dataset_name(self):
        cfg = _cfg(data={"name": "nope"})
        with pytest.raises(ConfigError, match="not a registered dataset"):
            validate_config(cfg)

    def test_required_keys_come_from_the_spec(self, toy_dataset):
        validate_config({**_cfg(), "data": {"name": toy_dataset.name}})
        cfg = {**_cfg(), "data": {"root": "/r"}}
        with pytest.raises(ConfigError, match="data.metadata is required"):
            validate_config(cfg)

    def test_loss_must_fit_the_task(self, toy_dataset):
        cfg = _cfg(training={"loss": "focal"})
        validate_config(cfg)                                  # multi-label: fine
        cfg["data"] = {"name": toy_dataset.name}
        with pytest.raises(ConfigError, match="training.loss"):
            validate_config(cfg)                              # multiclass: not valid

    def test_pos_weight_length(self):
        cfg = _cfg(training={"pos_weight": [1.0, 2.0]})
        with pytest.raises(ConfigError, match="19 classes"):
            validate_config(cfg)
        validate_config(_cfg(training={"pos_weight": [1.0] * 19}))

    def test_pos_weight_rejected_for_multiclass(self, toy_dataset):
        cfg = _cfg(data={"name": toy_dataset.name}, training={"pos_weight": [1.0, 1.0]})
        with pytest.raises(ConfigError, match="multi-label/binary"):
            validate_config(cfg)


class TestConfusionClassNames:
    def _dec(self, tmp_path, names):
        from src.training.decorators import ConfusionMatrixDecorator
        return ConfusionMatrixDecorator(object(), output_dir=str(tmp_path),
                                        csv_dir=str(tmp_path), class_names=names)

    def test_generic_names_without_a_dataset(self, tmp_path):
        assert self._dec(tmp_path, None)._names(3) == ["class_0", "class_1", "class_2"]

    def test_mismatch_raises(self, tmp_path):
        dec = self._dec(tmp_path, ["a", "b"])
        with pytest.raises(ValueError, match="2 class names for 3"):
            dec._names(3)


def test_model_accepts_input_channels():
    from src.models.vit import build_model
    model = build_model("resnet18", num_classes=4, pretrained=False, in_chans=1)
    assert model(torch.zeros(2, 1, 32, 32)).shape == (2, 4)


def test_dataset_names_must_be_lowercase():
    with pytest.raises(ValueError, match="lowercase"):
        _spec(name="ISIC")


def test_select_by_optimal_threshold_rejected_without_threshold(toy_dataset):
    cfg = _cfg(data={"name": toy_dataset.name}, training={"select_by": "f1_optimal"})
    with pytest.raises(ConfigError, match="f1_optimal"):
        validate_config(cfg)
    validate_config(_cfg(training={"select_by": "f1_optimal"}))   # multi-label: fine


def test_input_channels_seen_through_wrappers():
    from src.models.vit import build_model, input_channels
    model = build_model("resnet18", num_classes=2, pretrained=False, in_chans=1)
    assert input_channels(model) == 1
    assert input_channels(torch.nn.Module()) == 3      # unknown model → RGB default

    class _Wrapped(torch.nn.Module):                    # DDP-style .module
        def __init__(self, m):
            super().__init__()
            self.module = m
    assert input_channels(_Wrapped(model)) == 1


def test_model_analyzer_handles_single_channel_models():
    from src.benchmark.model_analyzer import ModelAnalyzer
    from src.models.vit import build_model
    model = build_model("resnet18", num_classes=2, pretrained=False, in_chans=1)
    info = ModelAnalyzer(model, "resnet18", torch.device("cpu")).analyze()
    assert info.flops_per_image_mflops > 0
