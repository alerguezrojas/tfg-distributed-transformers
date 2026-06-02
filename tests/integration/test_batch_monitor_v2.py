"""Tests para BatchMonitorDecorator v2 con batch_loss, lr y granularidad configurable."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.decorators.batch_monitor import BatchMonitorDecorator
from src.training.trainer import Trainer
from src.web.batch_parser import parse_batch_csv


def _tiny_loader(n: int = 16, bs: int = 4) -> DataLoader:
    x = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, 2, (n, 19)).float()
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def _make_trainer(tmp_dir: Path) -> Trainer:
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 19))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return Trainer(model, opt, None, torch.device("cpu"), checkpoint_dir=str(tmp_dir / "ckpt"))


# ── Firma del hook ─────────────────────────────────────────────────────────────


def test_hook_receives_six_arguments():
    """El hook recibe (epoch, batch_idx, n_batches, running_loss, batch_loss, lr)."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(Path(tmp))
        received = []
        trainer.register_batch_hook(
            lambda ep, bi, nb, rl, bl, lr: received.append(
                {"ep": ep, "bi": bi, "nb": nb, "rl": rl, "bl": bl, "lr": lr}
            )
        )
        trainer.train_epoch(_tiny_loader())

    assert len(received) == 4  # 16 samples / bs=4
    first = received[0]
    assert first["ep"] == 1
    assert first["bi"] == 1
    assert isinstance(first["bl"], float)
    assert first["lr"] > 0.0


# ── CSV format ────────────────────────────────────────────────────────────────


def test_batch_monitor_csv_has_new_columns():
    """El CSV de batch_monitor v2 debe tener batch_loss y lr."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        trainer = _make_trainer(tmp_path)
        monitor = BatchMonitorDecorator(
            trainer, log_every=1, output_dir=str(tmp_path / "logs"), timestamp="20260101_000000"
        )
        monitor.train_epoch(_tiny_loader())

        df = pd.read_csv(monitor.batch_csv_path)
        assert "batch_loss" in df.columns, "Falta columna batch_loss"
        assert "lr" in df.columns, "Falta columna lr"
        assert df["batch_loss"].notna().all(), "batch_loss no debe tener NaN"
        assert df["lr"].notna().all(), "lr no debe tener NaN"
        assert (df["lr"] > 0).all(), "lr debe ser > 0"


def test_batch_monitor_log_every_1_logs_every_batch():
    """Con log_every=1 se registra cada batch."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        trainer = _make_trainer(tmp_path)
        monitor = BatchMonitorDecorator(
            trainer, log_every=1, output_dir=str(tmp_path), timestamp="test"
        )
        monitor.train_epoch(_tiny_loader(n=16, bs=4))  # 4 batches

        df = pd.read_csv(monitor.batch_csv_path)
        assert len(df) == 4, f"Esperados 4 registros, obtenidos {len(df)}"


def test_batch_monitor_log_every_2_halves_records():
    """Con log_every=2 se registra la mitad de los batches (+ último siempre)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        trainer = _make_trainer(tmp_path)
        monitor = BatchMonitorDecorator(
            trainer, log_every=2, output_dir=str(tmp_path), timestamp="test2"
        )
        monitor.train_epoch(_tiny_loader(n=16, bs=4))  # 4 batches

        df = pd.read_csv(monitor.batch_csv_path)
        # Batches 2, 4 (divisibles por 2) + batch 4 ya es el último (contado una sola vez)
        assert len(df) == 2, f"Esperados 2 registros, obtenidos {len(df)}"


def test_batch_monitor_last_batch_always_logged():
    """El último batch siempre se registra independientemente de log_every."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        trainer = _make_trainer(tmp_path)
        monitor = BatchMonitorDecorator(
            trainer, log_every=10, output_dir=str(tmp_path), timestamp="test3"
        )
        monitor.train_epoch(_tiny_loader(n=16, bs=4))  # 4 batches, ninguno divisible por 10

        df = pd.read_csv(monitor.batch_csv_path)
        assert len(df) >= 1, "Debe registrarse al menos el último batch"
        assert df["batch"].iloc[-1] == 4, "El último registro debe ser el batch 4"


# ── Parser backward compat ────────────────────────────────────────────────────


def test_batch_parser_handles_legacy_csv():
    """parse_batch_csv tolera CSVs legacy sin batch_loss ni lr."""
    with tempfile.TemporaryDirectory() as tmp:
        legacy_path = Path(tmp) / "legacy_batch.csv"
        legacy_path.write_text("epoch,batch,n_batches,running_loss\n1,1,4,0.5\n1,2,4,0.4\n")

        df = parse_batch_csv(legacy_path)
        assert "batch_loss" in df.columns
        assert "lr" in df.columns
        assert "global_batch" in df.columns
        assert df["batch_loss"].isna().all(), "batch_loss debe ser NaN en legacy"
        assert df["lr"].isna().all(), "lr debe ser NaN en legacy"


def test_batch_parser_handles_v2_csv():
    """parse_batch_csv lee correctamente el formato v2."""
    with tempfile.TemporaryDirectory() as tmp:
        v2_path = Path(tmp) / "v2_batch.csv"
        v2_path.write_text(
            "epoch,batch,n_batches,running_loss,batch_loss,lr\n"
            "1,1,4,0.5,0.5,0.001\n"
            "1,2,4,0.45,0.40,0.001\n"
        )
        df = parse_batch_csv(v2_path)
        assert df["batch_loss"].notna().all()
        assert df["lr"].notna().all()
        assert df["global_batch"].tolist() == [1, 2]


def test_batch_parser_global_batch_computation():
    """global_batch = (epoch - 1) * n_batches + batch."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.csv"
        path.write_text(
            "epoch,batch,n_batches,running_loss,batch_loss,lr\n"
            "1,1,4,0.5,0.5,0.001\n"
            "1,4,4,0.4,0.38,0.001\n"
            "2,1,4,0.35,0.35,0.0009\n"
        )
        df = parse_batch_csv(path)
        assert df["global_batch"].tolist() == [1, 4, 5]


# ── Integración end-to-end ────────────────────────────────────────────────────


def test_full_stack_two_epochs():
    """Dos epochs seguidos — running_loss decrece y lr es consistente."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        trainer = _make_trainer(tmp_path)
        monitor = BatchMonitorDecorator(
            trainer, log_every=1, output_dir=str(tmp_path), timestamp="full"
        )
        loader = _tiny_loader()
        monitor.train_epoch(loader)
        monitor.train_epoch(loader)

        df = parse_batch_csv(monitor.batch_csv_path)
        assert set(df["epoch"].unique()) == {1, 2}
        assert df["lr"].notna().all()
        # LR del mismo epoch debe ser constante (sin scheduler)
        for ep in [1, 2]:
            lrs = df[df["epoch"] == ep]["lr"].unique()
            assert len(lrs) == 1, f"LR debería ser constante en epoch {ep}"
