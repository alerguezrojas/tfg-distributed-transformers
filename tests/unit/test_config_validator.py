"""Unit tests for src/training/config_validator.py"""
import pytest

try:
    from src.training.config_validator import validate_config, ConfigError
except ImportError:
    pytest.skip("config_validator not available in this branch", allow_module_level=True)


def _base_cfg(**overrides):
    cfg = {
        "data": {"root": "/tmp/ds", "metadata": "/tmp/meta.parquet", "num_workers": 4},
        "model": {"name": "vit_tiny_patch16_224", "num_classes": 19},
        "training": {"epochs": 10, "batch_size": 32, "lr": 0.0001, "warmup_epochs": 3},
    }
    for k, v in overrides.items():
        cfg["training"][k] = v
    return cfg


class TestValidConfig:
    def test_minimal_valid(self):
        validate_config(_base_cfg())  # should not raise

    def test_full_valid(self):
        cfg = _base_cfg(
            weight_decay=0.05, lr_min=1e-6, grad_clip=1.0,
            label_smoothing=0.1, mixup_alpha=0.2,
            early_stopping_patience=10,
        )
        validate_config(cfg)

    def test_optional_fields_missing(self):
        cfg = {"data": {"root": "/x", "metadata": "/y"},
               "model": {"name": "vit_tiny", "num_classes": 19},
               "training": {"epochs": 5, "batch_size": 32, "lr": 0.0001}}
        validate_config(cfg)

    def test_valid_distributed(self):
        cfg = _base_cfg()
        cfg["distributed"] = {
            "backend": "nccl",
            "ranks": [{"device": "cuda", "batch_size": 64, "compute_weight": 16},
                      {"device": "cpu", "batch_size": 4, "compute_weight": 1}],
        }
        validate_config(cfg)


class TestMissingRequired:
    def test_missing_data_section(self):
        cfg = {"model": {"name": "x", "num_classes": 19},
               "training": {"epochs": 5, "batch_size": 32, "lr": 0.0001}}
        with pytest.raises(ConfigError, match="data"):
            validate_config(cfg)

    def test_missing_model_section(self):
        cfg = {"data": {"root": "/x", "metadata": "/y"},
               "training": {"epochs": 5, "batch_size": 32, "lr": 0.0001}}
        with pytest.raises(ConfigError, match="model"):
            validate_config(cfg)

    def test_missing_lr(self):
        cfg = _base_cfg()
        del cfg["training"]["lr"]
        with pytest.raises(ConfigError, match="lr"):
            validate_config(cfg)

    def test_missing_epochs(self):
        cfg = _base_cfg()
        del cfg["training"]["epochs"]
        with pytest.raises(ConfigError, match="epochs"):
            validate_config(cfg)


class TestTypeErrors:
    def test_string_lr(self):
        cfg = _base_cfg()
        cfg["training"]["lr"] = "1e-4"
        with pytest.raises(ConfigError, match="string"):
            validate_config(cfg)

    def test_string_weight_decay(self):
        cfg = _base_cfg()
        cfg["training"]["weight_decay"] = "5e-2"
        with pytest.raises(ConfigError, match="string"):
            validate_config(cfg)

    def test_negative_epochs(self):
        cfg = _base_cfg()
        cfg["training"]["epochs"] = -1
        with pytest.raises(ConfigError):
            validate_config(cfg)

    def test_zero_batch_size(self):
        cfg = _base_cfg()
        cfg["training"]["batch_size"] = 0
        with pytest.raises(ConfigError):
            validate_config(cfg)

    def test_invalid_num_classes(self):
        cfg = _base_cfg()
        cfg["model"]["num_classes"] = 0
        with pytest.raises(ConfigError, match="num_classes"):
            validate_config(cfg)

    def test_dropout_out_of_range(self):
        cfg = _base_cfg()
        cfg["model"] = {"name": "x", "num_classes": 19, "dropout": 1.5}
        with pytest.raises(ConfigError, match="dropout"):
            validate_config(cfg)


class TestWarmupVsEpochs:
    def test_warmup_equals_epochs_raises(self):
        cfg = _base_cfg()
        cfg["training"]["epochs"] = 5
        cfg["training"]["warmup_epochs"] = 5
        with pytest.raises(ConfigError, match="warmup_epochs"):
            validate_config(cfg)

    def test_warmup_greater_than_epochs_raises(self):
        cfg = _base_cfg()
        cfg["training"]["epochs"] = 3
        cfg["training"]["warmup_epochs"] = 5
        with pytest.raises(ConfigError, match="warmup_epochs"):
            validate_config(cfg)

    def test_warmup_less_than_epochs_ok(self):
        cfg = _base_cfg()
        cfg["training"]["epochs"] = 10
        cfg["training"]["warmup_epochs"] = 3
        validate_config(cfg)


class TestDistributedConfig:
    def test_invalid_backend(self):
        cfg = _base_cfg()
        cfg["distributed"] = {"backend": "invalid"}
        with pytest.raises(ConfigError, match="backend"):
            validate_config(cfg)

    def test_invalid_rank_device(self):
        cfg = _base_cfg()
        cfg["distributed"] = {"ranks": [{"device": "tpu"}]}
        with pytest.raises(ConfigError, match="device"):
            validate_config(cfg)

    def test_negative_compute_weight(self):
        cfg = _base_cfg()
        cfg["distributed"] = {"ranks": [{"device": "cuda", "compute_weight": -1}]}
        with pytest.raises(ConfigError, match="compute_weight"):
            validate_config(cfg)
