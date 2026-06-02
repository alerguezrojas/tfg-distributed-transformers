"""Tests that verify training decorators never write PNG files to disk.

El dashboard web genera todas las gráficas de forma interactiva desde los CSVs.
Los decoradores de training solo deben escribir CSVs, nunca PNGs.
"""
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.decorators.confusion import ConfusionMatrixDecorator
from src.training.decorators.plotting import PlottingDecorator
from src.training.trainer import Trainer


# ── helpers ───────────────────────────────────────────────────────────────────


def _tiny_loader(n_samples: int = 16, batch_size: int = 4, n_classes: int = 19):
    x = torch.randn(n_samples, 3, 224, 224)
    y = torch.randint(0, 2, (n_samples, n_classes)).float()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _tiny_model(n_classes: int = 19) -> nn.Module:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, n_classes),
    )


def _make_trainer(tmp_dir: Path) -> Trainer:
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        criterion=nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"),
        checkpoint_dir=str(tmp_dir / "ckpt"),
    )


# ── tests ─────────────────────────────────────────────────────────────────────


def test_plotting_decorator_produces_no_png():
    """PlottingDecorator solo acumula métricas en memoria, sin escribir archivos."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        trainer = _make_trainer(tmp_path)
        decorated = PlottingDecorator(trainer, output_path=str(tmp_path / "plots"))

        loader = _tiny_loader()
        decorated.train_epoch(loader)
        decorated.eval_epoch(loader)

        png_files = list(tmp_path.rglob("*.png"))
        assert png_files == [], (
            f"PlottingDecorator generó PNGs inesperados: {png_files}"
        )


def test_confusion_matrix_decorator_produces_no_png():
    """ConfusionMatrixDecorator escribe CSVs pero nunca PNGs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        plots_dir = tmp_path / "plots"
        logs_dir = tmp_path / "logs"
        plots_dir.mkdir()
        logs_dir.mkdir()

        trainer = _make_trainer(tmp_path)
        decorated = ConfusionMatrixDecorator(
            trainer,
            output_dir=str(plots_dir),
            timestamp="20260101_000000",
            csv_dir=str(logs_dir),
        )

        loader = _tiny_loader()
        decorated.train_epoch(loader)
        decorated.eval_epoch(loader)

        png_files = list(tmp_path.rglob("*.png"))
        assert png_files == [], (
            f"ConfusionMatrixDecorator generó PNGs inesperados: {png_files}"
        )


def test_confusion_matrix_decorator_writes_csvs():
    """ConfusionMatrixDecorator debe escribir perclass y confusion_matrix CSVs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        trainer = _make_trainer(tmp_path)
        decorated = ConfusionMatrixDecorator(
            trainer,
            output_dir=str(tmp_path / "plots"),
            timestamp="20260101_000000",
            csv_dir=str(logs_dir),
        )

        loader = _tiny_loader()
        decorated.train_epoch(loader)
        decorated.eval_epoch(loader)

        csv_files = list(logs_dir.glob("*.csv"))
        names = {p.name for p in csv_files}
        assert "perclass_metrics_20260101_000000.csv" in names
        assert "confusion_matrix_20260101_000000.csv" in names


def test_plotting_decorator_history_populated():
    """PlottingDecorator acumula métricas de train y val en _history."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(Path(tmp))
        decorated = PlottingDecorator(trainer)

        loader = _tiny_loader()
        decorated.train_epoch(loader)
        decorated.eval_epoch(loader)

        assert len(decorated._history) > 0, "PlottingDecorator._history debe tener entradas"
        assert any("train_" in k for k in decorated._history), "Debe haber claves train_*"
        assert any("val_" in k for k in decorated._history), "Debe haber claves val_*"


def test_no_png_in_full_stack():
    """Stack completo (PlottingDecorator + ConfusionMatrixDecorator) no genera PNGs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        trainer = _make_trainer(tmp_path)
        with_confusion = ConfusionMatrixDecorator(
            trainer,
            output_dir=str(tmp_path / "plots"),
            timestamp="20260601_120000",
            csv_dir=str(tmp_path / "logs"),
        )
        with_plotting = PlottingDecorator(with_confusion, output_path=str(tmp_path / "plots"))

        loader = _tiny_loader()
        with_plotting.train_epoch(loader)
        with_plotting.eval_epoch(loader)

        png_files = list(tmp_path.rglob("*.png"))
        assert png_files == [], f"Stack completo generó PNGs: {png_files}"
