"""Instantánea de la línea ``Configuración:`` de cada estrategia, ejecutando los lanzadores reales.

La línea es un contrato con la web (``src/web/``) y con ``src/run_manifest.py``: su texto no puede
cambiar al refactorizar los lanzadores. Cada test lanza el script de la estrategia en proceso, en
CPU, con un dataset sintético registrado y un ViT diminuto, y compara la línea con el texto fijado.
"""
import runpy
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import TensorDataset

from src.data import registry
from src.data.registry import register_dataset
from src.data.spec import DatasetSpec

ROOT = Path(__file__).resolve().parents[2]
DS_NAME = "toy_launch"


def _toy_build(data_cfg, split, transform):
    g = torch.Generator().manual_seed(0 if split == "train" else 1)
    x = torch.randn(24, 3, 32, 32, generator=g)
    y = (torch.rand(24, 3, generator=g) < 0.4).float()
    return TensorDataset(x, y)


@pytest.fixture
def toy_dataset():
    register_dataset(DatasetSpec(
        name=DS_NAME, description="synthetic", classes=("a", "b", "c"), task="multilabel",
        in_channels=3, required_keys=(), build=_toy_build, transforms=lambda split: None))
    yield
    registry._REGISTRY.pop(DS_NAME, None)


def _write_config(tmp_path: Path, **extra) -> Path:
    cfg = {
        "data": {"name": DS_NAME, "num_workers": 0},
        "model": {"name": "vit_tiny_patch16_224", "pretrained": False, "img_size": 32},
        "training": {"epochs": 1, "batch_size": 8, "lr": 0.001, "seed": 3, "precision": "fp32"},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
        "output": {"env": "snap"},
        **extra,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _config_line(tmp_path: Path) -> str:
    logs = sorted((tmp_path / "logs").rglob("train_*.log"))
    assert len(logs) == 1, logs
    lines = [ln for ln in logs[0].read_text(encoding="utf-8").splitlines() if "Configuración:" in ln]
    assert len(lines) == 1, lines
    return lines[0].split("Configuración:", 1)[1].strip()


def _run_script(script: str, argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [script, *argv])
    runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")


@pytest.fixture
def rank0_env(monkeypatch):
    """Variables que inyecta torchrun, para un grupo gloo de un solo proceso."""
    for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29477"}.items():
        monkeypatch.setenv(key, value)
    yield
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


_TAIL = ("epochs=1 | lr=0.001 | precision=fp32 | loss=bce | train=24 | val=24 | "
         "dataset=toy_launch | seed=3 | ")


def test_single_config_line(tmp_path, monkeypatch, toy_dataset):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path)
    _run_script("train_single_gpu.py", ["--config", str(cfg), "--trace", "simple"], monkeypatch)
    line = _config_line(tmp_path)
    assert line.startswith("modelo=vit_tiny_patch16_224 | batch=8/GPU (global=8) | " + _TAIL)


def test_ddp_config_line(tmp_path, monkeypatch, toy_dataset, rank0_env):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, distributed={"backend": "gloo"})
    _run_script("train_ddp.py", ["--config", str(cfg), "--trace", "simple"], monkeypatch)
    line = _config_line(tmp_path)
    assert line.startswith(
        "modelo=vit_tiny_patch16_224 | batch=8/GPU (global=8, 1 GPUs) | " + _TAIL)


def test_model_parallel_config_line(tmp_path, monkeypatch, toy_dataset):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path)
    _run_script("train_model_parallel.py",
                ["--config", str(cfg), "--devices", "cpu,cpu", "--split-block", "1"], monkeypatch)
    line = _config_line(tmp_path)
    assert line.startswith(
        "modelo=vit_tiny_patch16_224 | paralelismo=modelo(pipeline) | devices=cpu+cpu | "
        "split_block=1 | batch=8 (global) | " + _TAIL)


def test_heterogeneous_config_line(tmp_path, monkeypatch, toy_dataset, rank0_env):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, distributed={
        "backend": "gloo", "ranks": [{"device": "cpu", "batch_size": 8, "compute_weight": 1}]})
    _run_script("train_heterogeneous_ddp.py", ["--config", str(cfg), "--trace", "simple"],
                monkeypatch)
    line = _config_line(tmp_path)
    assert line.startswith(
        "modelo=vit_tiny_patch16_224 | batch=GPU 8 / CPU — (global=8) | "
        "reparto=GPU 100% (~24 img) / CPU 0% (~0 img) | epochs=1 | lr=0.001 | loss=bce | "
        "train=24 | val=24 | dataset=toy_launch | seed=3 | ")


# ── Flujo común del lanzador ───────────────────────────────────────────────────

def _run_main(strategy: str, argv: list[str]) -> None:
    from src.training.launch import main
    main(strategy, argv)


def test_resume_trains_only_the_remaining_epochs(tmp_path, monkeypatch, toy_dataset, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path)
    _run_main("single", ["--config", str(cfg), "--epochs", "1"])
    last = next((tmp_path / "ckpt").rglob("last.pt"))
    capsys.readouterr()
    _run_main("single", ["--config", str(cfg), "--epochs", "2", "--resume", str(last)])
    assert "Resumiendo desde epoch 1 — quedan 1 epochs" in capsys.readouterr().out


def test_resume_works_in_ddp(tmp_path, monkeypatch, toy_dataset, rank0_env, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, distributed={"backend": "gloo"})
    _run_main("ddp", ["--config", str(cfg), "--epochs", "1"])
    last = next((tmp_path / "ckpt").rglob("last.pt"))
    capsys.readouterr()
    _run_main("ddp", ["--config", str(cfg), "--epochs", "2", "--resume", str(last)])
    assert "Resumiendo desde epoch 1 — quedan 1 epochs" in capsys.readouterr().out


def test_distributed_strategy_without_torchrun_says_so(tmp_path, monkeypatch, toy_dataset):
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    cfg = _write_config(tmp_path, distributed={"backend": "gloo"})
    for strategy in ("ddp", "heterogeneous"):
        with pytest.raises(SystemExit, match="torchrun"):
            _run_main(strategy, ["--config", str(cfg)])


def test_batch_size_and_precision_overrides_reach_the_config_line(tmp_path, monkeypatch,
                                                                  toy_dataset):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path)
    _run_main("single", ["--config", str(cfg), "--batch-size", "4", "--precision", "bf16"])
    assert "batch=4/GPU (global=4) | epochs=1 | lr=0.001 | precision=bf16" in _config_line(tmp_path)


def test_unsupported_option_stops_before_building_anything(tmp_path, monkeypatch, toy_dataset,
                                                           capsys):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, distributed={"backend": "gloo"})
    with pytest.raises(SystemExit) as exc:
        _run_main("heterogeneous", ["--config", str(cfg), "--precision", "amp"])
    assert exc.value.code == 2
    assert "--precision" in capsys.readouterr().err
    assert not (tmp_path / "logs").exists()
