"""--seed / --data-root / --epochs: sobrescritura común y ajuste de warmup en los scripts."""
import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import pytest

from src.training.config_overrides import add_common_overrides, apply_common_overrides

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ["train_single_gpu", "train_ddp", "train_heterogeneous_ddp", "train_model_parallel"]


def _cfg(**train):
    return {"data": {"root": "/orig"}, "training": {"epochs": 10, "warmup_epochs": 1, **train}}


def _ns(**kw):
    base = dict(data_root=None, epochs=None, seed=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses_seed_and_data_root(name, monkeypatch):
    mod = _load_script(name)
    monkeypatch.setattr(sys, "argv", [name, "--seed", "7", "--data-root", "/d", "--epochs", "1"])
    args = mod.parse_args()
    assert (args.seed, args.data_root, args.epochs) == (7, "/d", 1)
    cfg = _cfg()
    apply_common_overrides(cfg, args)
    assert cfg["training"]["seed"] == 7 and cfg["data"]["root"] == "/d"


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_defaults_do_not_override(name, monkeypatch):
    mod = _load_script(name)
    monkeypatch.setattr(sys, "argv", [name])
    args = mod.parse_args()
    assert args.seed is None and args.data_root is None
    cfg = _cfg(seed=3)
    apply_common_overrides(cfg, args)
    assert cfg["training"]["seed"] == 3 and cfg["data"]["root"] == "/orig"


def test_seed_zero_is_applied():
    cfg = _cfg(seed=5)
    apply_common_overrides(cfg, _ns(seed=0))
    assert cfg["training"]["seed"] == 0


def test_add_common_overrides_registers_options():
    p = argparse.ArgumentParser()
    add_common_overrides(p)
    a = p.parse_args(["--seed", "1", "--data-root", "x"])
    assert a.seed == 1 and a.data_root == "x"


def test_epochs_one_lowers_warmup_to_zero(caplog):
    cfg = _cfg()
    with caplog.at_level(logging.WARNING):
        apply_common_overrides(cfg, _ns(epochs=1))
    assert cfg["training"]["epochs"] == 1 and cfg["training"]["warmup_epochs"] == 0
    assert "warmup_epochs" in caplog.text


def test_epochs_lowers_warmup_to_epochs_minus_one():
    cfg = _cfg(warmup_epochs=5)
    apply_common_overrides(cfg, _ns(epochs=3))
    assert cfg["training"]["warmup_epochs"] == 2


def test_epochs_keeps_warmup_when_valid(caplog):
    cfg = _cfg(warmup_epochs=1)
    with caplog.at_level(logging.WARNING):
        apply_common_overrides(cfg, _ns(epochs=2))
    assert cfg["training"]["warmup_epochs"] == 1 and not caplog.text


def test_without_epochs_override_yaml_is_untouched():
    # Un YAML con warmup >= epochs no se corrige aquí: lo rechaza el validador.
    cfg = _cfg(epochs=1, warmup_epochs=1)
    apply_common_overrides(cfg, _ns())
    assert cfg["training"]["warmup_epochs"] == 1
