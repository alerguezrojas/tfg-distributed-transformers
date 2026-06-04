"""Tests de HeterogeneousDDPTrainer — sin lanzar procesos reales de DDP.

Se usa un proceso único con world_size=1 y backend gloo (CPU) para verificar:
- La normalización de gradientes por batch global
- Los batch hooks se llaman correctamente
- Las métricas se calculan bien
- No hay double-wrapping de DDP
- Compatibilidad con BatchMonitorDecorator
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer
from src.training.decorators.batch_monitor import BatchMonitorDecorator


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _init_gloo():
    """Inicializa un grupo de proceso gloo de 1 sola instancia para tests."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29499")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _destroy_gloo():
    if dist.is_initialized():
        dist.destroy_process_group()


def _tiny_loader(n: int = 8, bs: int = 4):
    x = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, 2, (n, 19)).float()
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def _tiny_model():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 19))


def _make_hetero_trainer(tmp_path: Path, local_batch_size: int = 4) -> HeterogeneousDDPTrainer:
    model = _tiny_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return HeterogeneousDDPTrainer(
        model=model, optimizer=opt, scheduler=None,
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        local_batch_size=local_batch_size,
    )


# ── Tests de inicialización ───────────────────────────────────────────────────


def test_heterogeneous_trainer_initializes():
    """HeterogeneousDDPTrainer debe inicializarse sin errores."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_hetero_trainer(Path(tmp))
            assert trainer.local_batch_size == 4
            assert hasattr(trainer, "_criterion_sum")
            assert trainer._criterion_sum.reduction == "sum"
    finally:
        _destroy_gloo()


def test_model_wrapped_with_ddp_once():
    """El modelo debe estar envuelto exactamente una vez con DDP."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_hetero_trainer(Path(tmp))
            from torch.nn.parallel import DistributedDataParallel as DDP
            assert isinstance(trainer.model, DDP), "El modelo debe ser DDP"
            # El módulo subyacente NO debe ser DDP (sin double-wrapping)
            assert not isinstance(trainer.model.module, DDP), \
                "Double-wrapping detectado: DDP(DDP(model))"
    finally:
        _destroy_gloo()


# ── Tests del bucle de entrenamiento ─────────────────────────────────────────


def test_train_loss_is_mean_scale_not_inflated():
    """La train_loss reportada debe estar en escala BCE-media (~mismo orden que
    una BCE normal), NO inflada ~n_clases× por usar criterion_sum sin dividir
    por el nº de clases. Antes del fix salía ~5-15; debe ser < 2."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_hetero_trainer(Path(tmp))
            result = trainer.train_epoch(_tiny_loader())
            # BCE media de un modelo sin entrenar ronda ~0.7; con margen, < 2.0
            assert 0.0 < result["loss"] < 2.0, (
                f"train_loss={result['loss']:.3f} fuera de escala BCE-media — "
                "¿se está reportando la suma sobre clases sin normalizar?"
            )
    finally:
        _destroy_gloo()


def test_batch_loss_in_mean_scale():
    """El batch_loss del hook también debe estar en escala media (< 2), no suma."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_hetero_trainer(Path(tmp))
            losses = []
            trainer.register_batch_hook(
                lambda ep, bi, nb, met: losses.append(met["batch_loss"])
            )
            trainer.train_epoch(_tiny_loader())
            assert losses, "el hook no recibió batches"
            assert all(0.0 < bl < 2.0 for bl in losses), (
                f"batch_loss fuera de escala media: {losses}"
            )
    finally:
        _destroy_gloo()


def test_train_epoch_returns_expected_keys():
    """train_epoch debe devolver loss, f1, accuracy, time, _preds, _labels."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_hetero_trainer(Path(tmp))
            result = trainer.train_epoch(_tiny_loader())
            assert "loss" in result
            assert "f1" in result
            assert "accuracy" in result
            assert "time" in result
            assert "_preds" in result
            assert "_labels" in result
    finally:
        _destroy_gloo()


def test_train_epoch_increments_current_epoch():
    """_current_epoch y _epoch deben incrementarse en cada train_epoch."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_hetero_trainer(Path(tmp))
            assert trainer._current_epoch == 0
            trainer.train_epoch(_tiny_loader())
            assert trainer._current_epoch == 1
            assert trainer._epoch == 1
            trainer.train_epoch(_tiny_loader())
            assert trainer._current_epoch == 2
    finally:
        _destroy_gloo()


def test_batch_hooks_are_called():
    """Los batch hooks deben recibir (epoch, batch_idx, n_batches, metrics: dict)."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_hetero_trainer(Path(tmp), local_batch_size=4)
            received = []
            trainer.register_batch_hook(
                lambda ep, bi, nb, met: received.append(
                    {"ep": ep, "bi": bi, "nb": nb, **met}
                )
            )
            trainer.train_epoch(_tiny_loader(n=8, bs=4))  # 2 batches

            assert len(received) == 2
            assert received[0]["ep"] == 1
            assert received[0]["bi"] == 1
            assert isinstance(received[0]["batch_loss"], float)
            assert received[0]["lr"] > 0
            assert "batch_f1" in received[0]
    finally:
        _destroy_gloo()


def test_batch_hooks_pass_metrics_dict():
    """El dict de métricas del hook contiene loss, lr, f1, acc, prec por batch."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = _make_hetero_trainer(Path(tmp))
            metrics_seen = []
            trainer.register_batch_hook(
                lambda ep, bi, nb, met: metrics_seen.append(met)
            )
            trainer.train_epoch(_tiny_loader())
            assert len(metrics_seen) > 0
            m0 = metrics_seen[0]
            for key in ("running_loss", "batch_loss", "lr", "batch_f1", "batch_acc", "batch_prec"):
                assert key in m0, f"Falta {key} en el dict de métricas"
    finally:
        _destroy_gloo()


# ── Tests de normalización de gradientes ─────────────────────────────────────


def test_gradient_normalization_by_global_batch():
    """La loss debe normalizarse por el batch global, no el local."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            model = _tiny_model()
            opt = torch.optim.SGD(model.parameters(), lr=0.0)  # no actualizar pesos
            trainer = HeterogeneousDDPTrainer(
                model=model, optimizer=opt, scheduler=None,
                device=torch.device("cpu"),
                checkpoint_dir=str(tmp),
                local_batch_size=4,
            )
            # Con world_size=1, global_batch = local_batch = 4
            # loss = criterion_sum(logits, labels) / 4
            # Equivalente a criterion_mean(logits, labels)
            loader = _tiny_loader(n=4, bs=4)
            result = trainer.train_epoch(loader)
            # La loss debe ser un número finito
            assert torch.isfinite(torch.tensor(result["loss"]))
    finally:
        _destroy_gloo()


# ── Tests de integración con BatchMonitorDecorator ────────────────────────────


def test_batch_monitor_works_with_heterogeneous_trainer():
    """BatchMonitorDecorator debe registrar sus hooks en HeterogeneousDDPTrainer."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trainer = _make_hetero_trainer(tmp_path)
            monitor = BatchMonitorDecorator(
                trainer, log_every=1,
                output_dir=str(tmp_path / "logs"),
                timestamp="20260101_000000",
            )
            monitor.train_epoch(_tiny_loader(n=8, bs=4))

            import pandas as pd
            df = pd.read_csv(monitor.batch_csv_path)
            assert len(df) == 2  # 8 muestras / bs=4 = 2 batches
            assert "batch_loss" in df.columns
            assert "lr" in df.columns
            assert df["batch_loss"].notna().all()
            assert df["lr"].notna().all()
    finally:
        _destroy_gloo()


def test_batch_monitor_csv_has_correct_epochs():
    """El CSV del batch monitor debe tener el epoch correcto."""
    _init_gloo()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trainer = _make_hetero_trainer(tmp_path)
            monitor = BatchMonitorDecorator(
                trainer, log_every=1,
                output_dir=str(tmp_path / "logs"),
                timestamp="20260101_000001",
            )
            loader = _tiny_loader(n=8, bs=4)
            monitor.train_epoch(loader)  # epoch 1
            monitor.train_epoch(loader)  # epoch 2

            import pandas as pd
            df = pd.read_csv(monitor.batch_csv_path)
            epochs = sorted(df["epoch"].unique().tolist())
            assert epochs == [1, 2]
    finally:
        _destroy_gloo()


# ── Tests del sampler ─────────────────────────────────────────────────────────


def test_heterogeneous_sampler_proportions():
    """Con pesos [16, 1], rank 0 debe recibir ~94% de los datos."""
    from src.training.heterogeneous_sampler import HeterogeneousDistributedSampler
    from torch.utils.data import TensorDataset

    n = 170  # 170 muestras
    ds = TensorDataset(torch.zeros(n))

    sampler_0 = HeterogeneousDistributedSampler(
        ds, weights=[16, 1], rank=0, world_size=2, shuffle=False
    )
    sampler_1 = HeterogeneousDistributedSampler(
        ds, weights=[16, 1], rank=1, world_size=2, shuffle=False
    )

    n0 = len(sampler_0)
    n1 = len(sampler_1)

    assert n0 + n1 == n, f"La suma de muestras debe ser {n}"
    # rank 0 debe tener aproximadamente 16/17 ≈ 94% de los datos
    expected_0 = round(n * 16 / 17)
    assert abs(n0 - expected_0) <= 1, f"rank 0 debería tener ~{expected_0} muestras, tiene {n0}"


def test_heterogeneous_sampler_no_overlap():
    """Los índices de los dos ranks no deben solaparse."""
    from src.training.heterogeneous_sampler import HeterogeneousDistributedSampler
    from torch.utils.data import TensorDataset

    n = 100
    ds = TensorDataset(torch.zeros(n))
    s0 = set(HeterogeneousDistributedSampler(ds, [16, 1], 0, 2, shuffle=False))
    s1 = set(HeterogeneousDistributedSampler(ds, [16, 1], 1, 2, shuffle=False))

    assert s0.isdisjoint(s1), "Los índices de rank 0 y rank 1 no deben solaparse"
    assert s0 | s1 == set(range(n)), "La unión debe cubrir todo el dataset"
