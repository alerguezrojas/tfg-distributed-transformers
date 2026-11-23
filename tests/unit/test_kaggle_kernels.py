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


@pytest.mark.parametrize("name,base,overrides",
                         [e for batch in train.BATCHES.values() for e in batch])
def test_every_experiment_config_is_valid(name, base, overrides):
    cfg = train.build_config(REPO, name, base, overrides)
    validate_config(cfg)
    assert cfg["data"]["image_size"] == cfg["model"]["img_size"]
    assert cfg["training"]["batch_size"] * 2 == 16            # global batch of the local runs
    assert cfg["checkpoint"]["dir"].endswith(name)            # one folder per experiment


def test_experiment_names_are_unique_across_batches():
    names = [name for batch in train.BATCHES.values() for name, _, _ in batch]
    assert len(names) == len(set(names))                      # results/<name> never overwritten
    assert train.EXPERIMENTS is train.BATCHES[train.BATCH]


def test_deep_update_merges_nested_dicts():
    base = {"data": {"a": 1, "b": 2}, "x": 1}
    assert train.deep_update(base, {"data": {"b": 3}}) == {"data": {"a": 1, "b": 3}, "x": 1}


def test_metadata_does_not_commit_the_username():
    for meta in REPO.glob("kaggle/*/kernel-metadata.json"):
        text = meta.read_text()
        assert '"{user}/' in text and "alu0" not in text


bench = _load("kaggle/benchmark_t4/run.py")


def test_benchmark_kernel_parses_the_report():
    report = """  Batch  Modo      imgs/s(train)  imgs/s(eval)     VRAM
      8  off             138.6       471.7   1.96 GB
     16  off             153.6       531.7   3.86 GB
     16  simple          150.0       500.0   3.90 GB
     16  off                 OOM
"""
    rows = bench.parse_rows(report)
    assert rows == [{"batch": 8, "train_ips": 138.6, "eval_ips": 471.7, "vram_gb": 1.96},
                    {"batch": 16, "train_ips": 153.6, "eval_ips": 531.7, "vram_gb": 3.86}]
    assert bench.parse_rows("     16  off                 OOM\n") == [
        {"batch": 16, "train_ips": None, "eval_ips": None, "vram_gb": None}]
    assert len(set(bench.MODELS)) == len(bench.MODELS) == 13
