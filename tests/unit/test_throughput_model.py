"""Two-term throughput model (compute + saved activations) and its validation helpers."""
import numpy as np
import pytest

from src.analysis.throughput_model import (
    common_factor,
    fit_flops_only,
    fit_two_term,
    leave_one_out,
    predict_ips,
    relative_errors,
    shape_errors,
    transfer,
)

A_TRUE, B_TRUE = 2e-4, 1.5e-2
F = np.array([30.0, 100.0, 25.0, 2.0, 15.0, 5.0])
S = np.array([0.40, 0.95, 0.70, 0.46, 0.80, 0.89])
IPS = 1.0 / (A_TRUE * F + B_TRUE * S)


def test_fit_recovers_exact_coefficients():
    a, b = fit_two_term(F, S, IPS)
    assert a == pytest.approx(A_TRUE) and b == pytest.approx(B_TRUE)
    assert predict_ips(a, b, F, S) == pytest.approx(IPS)


def test_flops_only_misses_memory_bound_models():
    a = fit_flops_only(F, IPS)
    err = relative_errors(predict_ips(a, 0.0, F, S), IPS)
    assert err[3] > 1.0            # the 2-GFLOP, memory-heavy model: far too fast
    assert np.all(np.abs(leave_one_out(F, S, IPS)) < 1e-9)    # exact data: exact LOO


def test_leave_one_out_flops_only_has_errors():
    assert np.abs(leave_one_out(F, S, IPS, two_term=False)).max() > 0.5


def test_fit_needs_two_points():
    with pytest.raises(ValueError):
        fit_two_term([1.0], [1.0], [1.0])


def test_transfer_scales_each_term():
    assert transfer(1.0, 2.0, 1.5, 1.4) == pytest.approx((1.5, 2.8))
    with pytest.raises(ValueError):
        transfer(1.0, 1.0, 0.0, 1.0)


def test_common_factor_and_shape():
    e = np.array([0.5, 0.5, 0.5])                   # every model 1.5× too fast
    assert common_factor(e) == pytest.approx(1.5)
    assert shape_errors(e) == pytest.approx([0.0, 0.0, 0.0])
    e2 = np.array([1.0, 0.0])                       # 2× and 1×: geometric mean √2
    assert common_factor(e2) == pytest.approx(np.sqrt(2))
    assert shape_errors(e2) == pytest.approx([np.sqrt(2) - 1, 1 / np.sqrt(2) - 1])
