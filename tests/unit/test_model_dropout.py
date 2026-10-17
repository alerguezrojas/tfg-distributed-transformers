"""model.dropout from the config must reach the classification head.

Regression test: until Oct 2026 the builder never passed it on, so every run used
the 0.1 default even when the config asked for 0.3 (the "v3" configs).
"""
import pytest
import torch

from src.training.builder import TrainingSessionBuilder


def _head_dropout(tmp_path, monkeypatch, model_cfg):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "data": {"root": "/r", "metadata": "/m.parquet"},
        "model": {"name": "resnet18", "pretrained": False, **model_cfg},
        "training": {"epochs": 1, "batch_size": 4, "lr": 0.001},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
    }
    trainer = TrainingSessionBuilder(cfg, torch.device("cpu")).with_trace("off").build()
    return trainer.model.head[0].p


@pytest.mark.parametrize("p", [0.0, 0.3, 0.5])
def test_config_dropout_reaches_the_head(tmp_path, monkeypatch, p):
    assert _head_dropout(tmp_path, monkeypatch, {"dropout": p}) == p


def test_default_dropout_unchanged(tmp_path, monkeypatch):
    assert _head_dropout(tmp_path, monkeypatch, {}) == 0.1
