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


def test_vit_base_fits_b32_on_3060ti():
    m, g = model_spec("vit_base_patch16_224"), gpu_spec("RTX 3060 Ti")
    assert fits_in_memory(m, g, 32, "fp32")              # real 4.95 GB ≤ 8
    assert max_batch(m, g, "fp32") >= 32


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
