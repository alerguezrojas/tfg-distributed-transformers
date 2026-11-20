"""The Kaggle kernels build valid configs from the repo ones (a broken YAML would only show
up hours later on Kaggle)."""
import importlib.util
from pathlib import Path

import pytest

from src.training.config_validator import validate_config

REPO = Path(__file__).resolve().parents[2]


def _load(rel: str):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = _load("kaggle/panda_train/run.py")


@pytest.mark.parametrize("name,base,overrides", train.EXPERIMENTS)
def test_every_experiment_config_is_valid(name, base, overrides):
    cfg = train.build_config(REPO, name, base, overrides)
    validate_config(cfg)
    assert cfg["data"]["image_size"] == cfg["model"]["img_size"]
    assert cfg["training"]["batch_size"] * 2 == 16            # global batch of the local runs
    assert cfg["checkpoint"]["dir"].endswith(name)            # one folder per experiment


def test_deep_update_merges_nested_dicts():
    base = {"data": {"a": 1, "b": 2}, "x": 1}
    assert train.deep_update(base, {"data": {"b": 3}}) == {"data": {"a": 1, "b": 3}, "x": 1}


def test_metadata_does_not_commit_the_username():
    for meta in REPO.glob("kaggle/*/kernel-metadata.json"):
        text = meta.read_text()
        assert '"{user}/' in text and "alu0" not in text
