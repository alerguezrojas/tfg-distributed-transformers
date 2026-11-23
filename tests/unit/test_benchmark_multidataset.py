"""El benchmark no asume BigEarthNet: tamaños de split, perfil de I/O, predicción y resolución."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.benchmark import BenchmarkChecker, ReportFormatter
from src.benchmark.dataset_sizes import (BIGEARTHNET_TRAIN, BIGEARTHNET_VAL,
                                         resolve_split_sizes)


def _panda_root(tmp_path, n_train=7, n_val=3):
    n = n_train + n_val
    ids = [f"img{i}" for i in range(n)]
    pd.DataFrame({"image_id": ids, "data_provider": ["karolinska"] * n,
                  "isup_grade": [i % 6 for i in range(n)]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"image_id": ids,
                  "split": ["train"] * n_train + ["val"] * n_val}).to_csv(tmp_path / "splits.csv", index=False)
    return {"data": {"name": "panda", "root": str(tmp_path)}}


def test_panda_sizes_come_from_the_partition(tmp_path):
    assert resolve_split_sizes(_panda_root(tmp_path)) == (7, 3)


def test_explicit_sizes_win(tmp_path):
    cfg = _panda_root(tmp_path)
    assert resolve_split_sizes(cfg, 100, 50) == (100, 50)
    assert resolve_split_sizes(cfg, train_size=100) == (100, 3)


def test_unreadable_non_bigearthnet_is_an_error_not_a_guess(tmp_path):
    cfg = {"data": {"name": "panda", "root": str(tmp_path / "no_existe")}}
    with pytest.raises(ValueError, match="--train-size"):
        resolve_split_sizes(cfg)
    assert resolve_split_sizes(cfg, 10, 5) == (10, 5)


def test_bigearthnet_keeps_its_defaults():
    assert resolve_split_sizes({}) == (BIGEARTHNET_TRAIN, BIGEARTHNET_VAL)
    cfg = {"data": {"name": "bigearthnet", "metadata": "/no/existe.parquet"}}
    assert resolve_split_sizes(cfg) == (237871, 122342)


def _run(cfg):
    checker = BenchmarkChecker("vit_tiny_patch16_224", [2], [1], ["off"], 100, 50,
                               profile_disk=False, analyze_ddp=False, config=cfg)
    report = checker.run()
    fmt = ReportFormatter()
    fmt.print(report)
    return report, "\n".join(fmt._lines)


def test_non_bigearthnet_has_no_f1_prior():
    cfg = {"data": {"name": "medmnist_breast"}, "model": {"img_size": 32}}
    report, text = _run(cfg)
    assert report.performance_prediction is None
    assert "Sin prior calibrado" in text and "medmnist_breast" in text
    assert "Val F1 esperado" not in text
    assert "Patches encontrados" not in text


def test_bigearthnet_still_predicts_f1():
    report, text = _run({"model": {"img_size": 32}})
    assert report.performance_prediction is not None
    assert "Val F1 esperado" in text
    assert "Sin prior calibrado" not in text


def test_benchmark_uses_model_img_size():
    from src.models.vit import build_model, input_size
    cfg = {"data": {"name": "medmnist_breast"}, "model": {"img_size": 64}}
    checker = BenchmarkChecker("vit_tiny_patch16_224", [2], [1], ["off"], 100, 50, config=cfg)
    model = build_model(model_name="vit_tiny_patch16_224", pretrained=False,
                        num_classes=checker._spec.num_classes, in_chans=checker._spec.in_channels,
                        img_size=checker._img_size)
    assert input_size(model) == 64
    assert checker._benchmarker(model, "fp32")._side == 64
