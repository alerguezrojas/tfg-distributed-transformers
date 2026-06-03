"""Tests de compatibilidad del DDP con el backend gloo (usado en Verode).

gloo NO soporta ReduceOp.AVG (solo NCCL). El cluster Verode corre gloo sobre
torch 2.7.1, donde all_reduce con AVG lanza:
    RuntimeError: Cannot use ReduceOp.AVG with Gloo

Estos tests evitan que vuelva a colarse un ReduceOp.AVG y validan que el
eval distribuido promedia el loss correctamente con world_size=1 (gloo).
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Test de no-regresión por inspección del fuente ────────────────────────────


def test_no_reduceop_avg_in_codebase():
    """Ningún módulo de training debe usar ReduceOp.AVG (incompatible con gloo)."""
    root = Path(__file__).resolve().parents[2] / "src" / "training"
    offenders = []
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "ReduceOp.AVG" in text:
            offenders.append(py.name)
    assert not offenders, (
        f"ReduceOp.AVG no es compatible con gloo (backend de Verode). "
        f"Usar SUM + división. Ficheros afectados: {offenders}"
    )


# ── Test funcional del eval distribuido con gloo (world_size=1) ───────────────


def _init_gloo():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29488")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _destroy_gloo():
    if dist.is_initialized():
        dist.destroy_process_group()


def _tiny_loader(n=16, bs=4):
    x = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, 2, (n, 19)).float()
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def _tiny_model():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 19))


def test_ddp_eval_epoch_runs_with_gloo():
    """DDPTrainer.eval_epoch debe completar con gloo sin el error de ReduceOp.AVG."""
    from src.training.ddp_trainer import DDPTrainer
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            model = _tiny_model()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = DDPTrainer(
                model=model, optimizer=opt, scheduler=None,
                device=torch.device("cpu"), checkpoint_dir=str(tmp),
                rank=0, world_size=1,
            )
            result = trainer.eval_epoch(_tiny_loader())
            # Con world_size=1, SUM/1 == valor original → loss finito y válido
            assert "loss" in result
            assert torch.isfinite(torch.tensor(result["loss"]))
            assert "f1" in result and "accuracy" in result
    finally:
        _destroy_gloo()


def test_heterogeneous_eval_epoch_runs_with_gloo():
    """HeterogeneousDDPTrainer hereda el eval — también debe ir con gloo."""
    from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            model = _tiny_model()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = HeterogeneousDDPTrainer(
                model=model, optimizer=opt, scheduler=None,
                device=torch.device("cpu"), checkpoint_dir=str(tmp),
                rank=0, world_size=1, local_batch_size=4,
            )
            result = trainer.eval_epoch(_tiny_loader())
            assert torch.isfinite(torch.tensor(result["loss"]))
    finally:
        _destroy_gloo()
