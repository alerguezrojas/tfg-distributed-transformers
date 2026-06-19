"""Unit tests for src/performance_model.py — the analytic prediction engine.

Validates the closed-form model against the REAL Kaggle 2×T4 measurements
(documented in CLAUDE.md / the feasibility brief) and the limit regimes.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.performance_model import (
    estimate_rc, estimate_rio, estimate_vram_gb, fits_in_memory, max_batch,
    gpu_spec, model_spec, predict, predict_epoch,
    expected_best_f1, predict_quality, N_FULL_TRAIN,
)


# ── Parameter estimation ─────────────────────────────────────────────────────────

def test_rc_vit_base_t4_matches_measured():
    """vit_base on a T4 fp32 measured ≈ 26 img/s — the MFU calibration anchor."""
    rc = estimate_rc(model_spec("vit_base_patch16_224"), gpu_spec("Tesla T4"), "fp32")
    assert 22 <= rc <= 30


def test_precision_multiplies_compute():
    m, g = model_spec("vit_base_patch16_224"), gpu_spec("Tesla T4")
    assert estimate_rc(m, g, "amp") == pytest.approx(estimate_rc(m, g, "fp32") * 3.8, rel=0.01)


def test_fuzzy_gpu_and_model_lookup():
    assert gpu_spec("NVIDIA Tesla T4").name == "Tesla T4"
    assert gpu_spec("NVIDIA GeForce RTX 3060 Ti").name == "RTX 3060 Ti"
    assert model_spec("vit_base_patch16_224").params_m == 85.8
    assert gpu_spec("totally unknown gpu") is None


# ── Headline validations from the brief (predicted → real, <~10%) ────────────────

def _train_s(strategy, n, precision, batch=96):
    return predict(strategy, "vit_base_patch16_224", "Tesla T4", n_gpus=n,
                   dataset_size=5000, batch=batch, precision=precision,
                   epochs=15).time_per_epoch_train_s


def test_single_fp32_time_matches():
    assert _train_s("single", 1, "fp32") == pytest.approx(194, rel=0.10)   # real 194 s


def test_ddp_2gpu_speedup_matches():
    p = predict("ddp", "vit_base_patch16_224", "Tesla T4", n_gpus=2,
                dataset_size=5000, batch=96, precision="fp32", epochs=15)
    assert p.speedup == pytest.approx(1.96, rel=0.10)   # real 1.96×
    assert p.efficiency > 0.9
    assert p.bottleneck == "compute"


def test_amp_speedup_matches():
    fp32 = _train_s("single", 1, "fp32")
    amp = _train_s("single", 1, "amp")
    assert fp32 / amp == pytest.approx(3.80, rel=0.10)   # real 3.80×


def test_model_parallel_does_not_accelerate():
    p = predict("model_parallel", "vit_base_patch16_224", "Tesla T4", n_gpus=2,
                dataset_size=5000, batch=96, precision="fp32", epochs=15)
    assert p.speedup == pytest.approx(1.0, abs=0.1)      # real 1.02×
    assert any("does not accelerate" in n for n in p.notes)


# ── Limit regimes (the "bonito" cases for the report) ────────────────────────────

def test_vit_tiny_is_io_bound():
    """A tiny model on the same disk is I/O-bound → DDP barely helps."""
    p = predict("ddp", "vit_tiny_patch16_224", "Tesla T4", n_gpus=2,
                dataset_size=5000, batch=96, precision="fp32", epochs=15)
    assert p.bottleneck == "io"
    assert p.speedup < 1.4                                # real 1.27×


def test_vit_base_is_compute_bound():
    p = predict("ddp", "vit_base_patch16_224", "Tesla T4", n_gpus=2,
                dataset_size=5000, batch=96, precision="fp32", epochs=15)
    assert p.bottleneck == "compute"


def test_heterogeneous_penalizes():
    """V100 + CPU synchronous DDP runs slower than the GPU alone."""
    p = predict("heterogeneous", "vit_tiny_patch16_224", "Tesla V100", n_gpus=2,
                dataset_size=5000, batch=96, precision="fp32", epochs=3)
    assert p.speedup < 1.0


# ── Memory / OOM (validate the measured cases) ────────────────────────────────────

def test_vit_large_fits_b32_oom_b48_on_t4():
    m, g = model_spec("vit_large_patch16_224"), gpu_spec("Tesla T4")
    assert fits_in_memory(m, g, 32, "fp32")              # real 13.78 GB ≤ 16
    assert not fits_in_memory(m, g, 48, "fp32")          # real OOM


def test_vit_large_b32_vram_close_to_measured():
    v = estimate_vram_gb(model_spec("vit_large_patch16_224"), 32, "fp32")
    assert v == pytest.approx(13.78, rel=0.10)


def test_vit_base_fits_b32_oom_b64_on_3060ti():
    """The real RTX 3060 Ti behaviour: batch 32 fits (~4.95 GB), batch 64 OOMs."""
    m, g = model_spec("vit_base_patch16_224"), gpu_spec("RTX 3060 Ti")
    assert fits_in_memory(m, g, 32, "fp32")              # real 4.95 GB ≤ 8
    assert not fits_in_memory(m, g, 64, "fp32")          # real OOM
    assert max_batch(m, g, "fp32") == 32                 # not 64 (the old optimism)


def test_vit_base_vram_b32_close_to_measured():
    v = estimate_vram_gb(model_spec("vit_base_patch16_224"), 32, "fp32")
    assert v == pytest.approx(4.95, rel=0.10)            # measured on the 3060 Ti


def test_amp_uses_less_vram_than_fp32():
    m = model_spec("vit_base_patch16_224")
    assert estimate_vram_gb(m, 64, "amp") < estimate_vram_gb(m, 64, "fp32")


# ── Calibration hook ─────────────────────────────────────────────────────────────

def test_measured_rc_overrides_estimate():
    """Passing a real r_c calibrates the prediction (and flags it)."""
    p = predict("single", "vit_base_patch16_224", "Tesla T4", n_gpus=1,
                dataset_size=5000, batch=96, precision="fp32", epochs=1,
                rc_measured=50.0)
    assert p.calibrated
    # 5000 / 50 = 100 s with the forced r_c
    assert p.time_per_epoch_train_s == pytest.approx(100, rel=0.05)


def test_unknown_specs_return_none():
    assert predict("single", "nonexistent_model", "Tesla T4") is None
    assert predict("single", "vit_base_patch16_224", "nonexistent_gpu") is None


# ── Quality model: expected Val F1 vs dataset size (the honest prior) ─────────────

def test_quality_vit_base_full_matches_documented():
    """Full BigEarthNet-S2 → the documented ~0.68 plateau (v1–v4)."""
    f1, conf, _ = expected_best_f1("vit_base_patch16_224", N_FULL_TRAIN)
    assert f1 == pytest.approx(0.68, abs=0.01)
    assert conf == "high"


def test_quality_vit_base_subset_matches_kaggle():
    """5 000-image subset → ~0.55, the REAL Kaggle vit_base result (data-scaling)."""
    f1, _, _ = expected_best_f1("vit_base_patch16_224", 5000)
    assert f1 == pytest.approx(0.55, abs=0.02)


def test_quality_vit_tiny_subset_matches_real():
    """vit_tiny on the 5 000 subset measured ~0.27 (Kaggle/Verode)."""
    f1, _, _ = expected_best_f1("vit_tiny_patch16_224", 5000)
    assert f1 == pytest.approx(0.27, abs=0.04)


def test_quality_more_data_is_never_worse():
    big, _, _ = expected_best_f1("vit_base_patch16_224", 100_000)
    small, _, _ = expected_best_f1("vit_base_patch16_224", 5000)
    assert big > small


def test_quality_prediction_curve_and_fields():
    q = predict_quality("vit_base_patch16_224", dataset_size=5000, epochs=15)
    assert q.method == "empirical-prior"
    assert q.early_stop_epoch > q.best_epoch
    assert len(q.curve_val_f1) == len(q.curve_epochs)
    # learning curve rises early, train sits above val (the overfitting gap)
    assert q.curve_val_f1[4] > q.curve_val_f1[0]
    assert q.curve_train_f1[-1] >= q.curve_val_f1[-1]
    assert any("planning prior" in n for n in q.notes)


def test_quality_unknown_model_falls_back_to_vit_base():
    q = predict_quality("some_unknown_xyz", dataset_size=N_FULL_TRAIN)
    assert q.expected_best_f1 == pytest.approx(0.68, abs=0.01)
