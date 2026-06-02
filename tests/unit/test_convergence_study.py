"""Tests de ConvergenceStudy — ajuste de curvas (puro) y mediciones (modelo tiny)."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.convergence_study import (
    ConvergenceStudy,
    LRRangeResult, ConvergenceResult, GradientNoiseResult, StudyReport,
    fit_power_law, extrapolate_power_law, loss_to_f1_estimate,
)


# ── Ajuste de curvas (puro, sin GPU) ──────────────────────────────────────────


def test_fit_power_law_recovers_known_curve():
    """Genera loss(t)=a·t^-b+c y verifica que el fit recupera los parámetros."""
    a_true, b_true, c_true = 2.0, 0.5, 0.15
    steps = np.arange(1, 60)
    losses = a_true * steps ** (-b_true) + c_true
    a, b, c, r2 = fit_power_law(steps, losses)
    assert r2 > 0.98, f"R² bajo: {r2}"
    # La curva ajustada debe predecir bien en t=100
    pred = extrapolate_power_law(a, b, c, 100)
    true = a_true * 100 ** (-b_true) + c_true
    assert abs(pred - true) < 0.05


def test_fit_power_law_monotone_decreasing():
    """La curva ajustada debe ser decreciente."""
    steps = np.arange(1, 50)
    losses = 1.5 * steps ** (-0.4) + 0.2 + np.random.default_rng(0).normal(0, 0.01, len(steps))
    a, b, c, r2 = fit_power_law(steps, losses)
    early = extrapolate_power_law(a, b, c, 5)
    late = extrapolate_power_law(a, b, c, 500)
    assert early > late, "La loss extrapolada debe decrecer"


def test_fit_power_law_insufficient_data():
    """Con <4 puntos no crashea."""
    a, b, c, r2 = fit_power_law([1, 2], [0.5, 0.4])
    assert isinstance(a, float)


def test_extrapolate_power_law_at_zero():
    """step<=0 devuelve a+c sin dividir por cero."""
    val = extrapolate_power_law(2.0, 0.5, 0.1, 0)
    assert val == pytest.approx(2.1)


def test_loss_to_f1_monotone():
    """Menor loss → mayor F1."""
    f1_high_loss = loss_to_f1_estimate(0.30, "vit_base")
    f1_low_loss = loss_to_f1_estimate(0.16, "vit_base")
    assert f1_low_loss > f1_high_loss


def test_loss_to_f1_bounded():
    """F1 acotado por el techo de la familia."""
    f1 = loss_to_f1_estimate(0.10, "vit_tiny")
    assert 0 <= f1 <= 0.55  # techo vit_tiny


def test_loss_to_f1_family_ceilings():
    """ViT-Base tiene mayor techo que ViT-Tiny."""
    f1_base = loss_to_f1_estimate(0.14, "vit_base")
    f1_tiny = loss_to_f1_estimate(0.14, "vit_tiny")
    assert f1_base > f1_tiny


# ── Mediciones con modelo tiny en CPU ─────────────────────────────────────────


def _tiny_loader(n=64, bs=8):
    x = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, 2, (n, 19)).float()
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def _tiny_model():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 19))


def test_lr_range_test_runs():
    study = ConvergenceStudy(torch.device("cpu"), "vit_tiny")
    result = study.lr_range_test(_tiny_model(), _tiny_loader(), n_steps=10)
    assert isinstance(result, LRRangeResult)
    assert len(result.lrs) == len(result.losses)
    assert len(result.lrs) <= 10
    assert result.suggested_lr > 0
    assert result.min_loss_lr > 0


def test_lr_range_test_lrs_increasing():
    study = ConvergenceStudy(torch.device("cpu"), "vit_tiny")
    result = study.lr_range_test(_tiny_model(), _tiny_loader(),
                                 lr_min=1e-6, lr_max=1e-1, n_steps=8)
    # Los LRs probados deben ser crecientes (hasta divergencia)
    for i in range(1, len(result.lrs)):
        assert result.lrs[i] > result.lrs[i - 1]


def test_convergence_test_runs():
    study = ConvergenceStudy(torch.device("cpu"), "vit_tiny")
    result = study.convergence_test(
        _tiny_model(), _tiny_loader(), lr=1e-3, n_steps=15,
        batch_size=8, n_train_images=1000, n_epochs_target=5,
    )
    assert isinstance(result, ConvergenceResult)
    assert len(result.steps) == 15
    assert len(result.losses) == 15
    assert len(result.f1s) == 15
    assert result.measured_imgs_per_s > 0
    assert 0 <= result.extrapolated_best_f1 <= 1
    assert result.extrapolated_loss_final >= 0


def test_convergence_extrapolates_below_initial():
    """La loss extrapolada a 1 epoch debe ser <= la loss inicial medida."""
    study = ConvergenceStudy(torch.device("cpu"), "vit_tiny")
    result = study.convergence_test(
        _tiny_model(), _tiny_loader(), lr=1e-2, n_steps=20,
        batch_size=8, n_train_images=500, n_epochs_target=5,
    )
    # Con muchos batches/epoch, loss_1ep debe ser <= primera loss
    assert result.extrapolated_loss_1ep <= result.losses[0] + 0.1


def test_gradient_noise_scale_runs():
    study = ConvergenceStudy(torch.device("cpu"), "vit_tiny")
    result = study.gradient_noise_scale(_tiny_model(), _tiny_loader(),
                                        n_batches=6, batch_size=8)
    assert isinstance(result, GradientNoiseResult)
    assert result.grad_norm_mean > 0
    assert result.grad_norm_std >= 0
    assert result.suggested_batch_size >= 8
    assert result.cv >= 0


def test_run_full_study():
    study = ConvergenceStudy(torch.device("cpu"), "vit_tiny")
    report = study.run_full_study(
        _tiny_model(), _tiny_loader(), lr=1e-3,
        batch_size=8, n_train_images=1000, n_epochs_target=5,
    )
    assert isinstance(report, StudyReport)
    assert report.lr_range is not None
    assert report.convergence is not None
    assert report.gradient_noise is not None
    assert "empírico" in report.notes.lower()


def test_run_full_study_can_skip_optional():
    study = ConvergenceStudy(torch.device("cpu"), "vit_tiny")
    report = study.run_full_study(
        _tiny_model(), _tiny_loader(), lr=1e-3,
        batch_size=8, n_train_images=1000, n_epochs_target=5,
        do_lr_range=False, do_gradient_noise=False,
    )
    assert report.lr_range is None
    assert report.gradient_noise is None
    assert report.convergence is not None
