"""Unit tests for src/cloud_cost.py — cloud training cost prediction."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cloud_cost import (
    CLOUD_OPTIONS, CloudOption, estimate_costs, gpu_tflops,
)


def test_gpu_tflops_fuzzy_match():
    assert gpu_tflops("Tesla T4") == 65.0
    assert gpu_tflops("NVIDIA Tesla V100-PCIE-32GB") == 112.0   # substring
    assert gpu_tflops("NVIDIA GeForce RTX 3060 Ti") == 35.0
    assert gpu_tflops("A100 80GB") == 312.0
    assert gpu_tflops("some unknown gpu") is None
    assert gpu_tflops(None) is None


def test_cost_is_hours_times_price_when_same_gpu():
    # Reference GPU == target GPU → no time scaling.
    opts = [CloudOption("X", "Tesla T4", 0.5)]
    rows = estimate_costs(total_hours_ref=10.0, ref_gpu_name="Tesla T4", options=opts)
    assert rows[0]["est_hours"] == 10.0
    assert rows[0]["cost_usd"] == pytest.approx(5.0)
    assert rows[0]["scaled"] is True


def test_faster_gpu_fewer_hours():
    # From a T4 (65) to an A100 (312): ~4.8× fewer hours.
    opts = [CloudOption("X", "A100 40GB", 4.0)]
    rows = estimate_costs(total_hours_ref=10.0, ref_gpu_name="Tesla T4", options=opts)
    assert rows[0]["est_hours"] == pytest.approx(10.0 * 65.0 / 312.0, abs=0.01)  # rounded to 2 dp
    assert rows[0]["est_hours"] < 10.0


def test_unknown_reference_falls_back_to_no_scaling():
    opts = [CloudOption("X", "Tesla T4", 0.5)]
    rows = estimate_costs(total_hours_ref=8.0, ref_gpu_name="mystery", options=opts)
    assert rows[0]["est_hours"] == 8.0
    assert rows[0]["scaled"] is False


def test_free_provider_zero_cost():
    rows = estimate_costs(total_hours_ref=5.0, ref_gpu_name="Tesla T4")
    free = [r for r in rows if r["usd_per_hour"] == 0.0]
    assert free, "expected at least one free option (Kaggle/Colab)"
    assert all(r["cost_usd"] == 0.0 for r in free)


def test_results_sorted_by_cost():
    rows = estimate_costs(total_hours_ref=20.0, ref_gpu_name="Tesla V100")
    costs = [r["cost_usd"] for r in rows]
    assert costs == sorted(costs)


def test_options_table_sane():
    assert len(CLOUD_OPTIONS) >= 8
    providers = {o.provider for o in CLOUD_OPTIONS}
    assert {"AWS", "GCP", "Kaggle"} <= providers
    for o in CLOUD_OPTIONS:
        assert o.usd_per_hour >= 0.0 and o.gpu
