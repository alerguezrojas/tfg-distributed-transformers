"""Pendiente/intercepto de calibración, ECE por clase y ECE adaptativo."""
import numpy as np
import pytest

from src.analysis.calibration import (adaptive_ece, calibration, calibration_curve_stats,
                                      per_class_calibration)


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _logit(p):
    return np.log(p / (1 - p))


def _simulate(a, b, n=40000, seed=0):
    """y ~ Bernoulli(sigmoid(a + b * logit(p))) con p uniforme: calibración verdadera (a, b)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, (n, 1))
    y = (rng.random((n, 1)) < _sigmoid(a + b * _logit(p))).astype(float)
    return p, y


# ------------------------------------------------------------------ pendiente e intercepto

def test_well_calibrated_model_has_slope_one_intercept_zero():
    p, y = _simulate(0.0, 1.0)
    s = calibration_curve_stats(p, y, "binary")
    assert s["converged"] and s["n"] == len(p)
    assert s["slope"] == pytest.approx(1.0, abs=4 * s["slope_se"])
    assert s["intercept"] == pytest.approx(0.0, abs=4 * s["intercept_se"])
    assert s["intercept_citl"] == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("a,b", [(0.5, 0.7), (-0.3, 1.4), (0.0, 0.4)])
def test_recovers_the_true_recalibration_parameters(a, b):
    p, y = _simulate(a, b, seed=1)
    s = calibration_curve_stats(p, y, "binary")
    assert s["slope"] == pytest.approx(b, abs=4 * s["slope_se"])
    assert s["intercept"] == pytest.approx(a, abs=4 * s["intercept_se"])
    assert 0 < s["slope_se"] < 0.05


def test_overconfident_has_slope_below_one_and_underconfident_above():
    p, y = _simulate(0.0, 0.5)         # la verdad es más moderada que p
    assert calibration_curve_stats(p, y, "binary")["slope"] < 0.6
    p, y = _simulate(0.0, 2.0)
    assert calibration_curve_stats(p, y, "binary")["slope"] > 1.8


def test_calibration_in_the_large_recovers_a_pure_shift():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, (30000, 1))
    y = (rng.random(p.shape) < _sigmoid(_logit(p) + 0.8)).astype(float)
    s = calibration_curve_stats(p, y, "binary")
    assert s["intercept_citl"] == pytest.approx(0.8, abs=0.05)
    assert s["slope"] == pytest.approx(1.0, abs=0.1)


def test_exact_solution_with_two_distinct_predictions():
    # Con dos valores de x el MLE reproduce las frecuencias: a + b x_i = logit(f_i).
    p = np.array([0.2] * 100 + [0.8] * 100)[:, None]
    y = np.array([1] * 30 + [0] * 70 + [1] * 60 + [0] * 40, dtype=float)[:, None]
    s = calibration_curve_stats(p, y, "binary", eps=1e-12)
    x1, x2 = _logit(0.2), _logit(0.8)
    f1, f2 = 0.3, 0.6
    slope = (_logit(f2) - _logit(f1)) / (x2 - x1)
    assert s["slope"] == pytest.approx(slope, rel=1e-6)
    assert s["intercept"] == pytest.approx(_logit(f1) - slope * x1, rel=1e-6)


def test_score_equations_hold_at_the_solution():
    p, y = _simulate(0.2, 0.8, n=5000, seed=3)
    s = calibration_curve_stats(p, y, "binary")
    x = _logit(np.clip(p[:, 0], 1e-6, 1 - 1e-6))
    mu = _sigmoid(s["intercept"] + s["slope"] * x)
    assert np.sum(y[:, 0] - mu) == pytest.approx(0, abs=1e-5)
    assert np.sum((y[:, 0] - mu) * x) == pytest.approx(0, abs=1e-5)


def test_multiclass_top_label_uses_max_prob_and_correctness():
    rng = np.random.default_rng(4)
    n, c = 20000, 3
    p = rng.dirichlet(np.ones(c), n)
    conf = p.max(axis=1)
    correct = rng.random(n) < _sigmoid(_logit(np.clip(conf, 1e-6, 1 - 1e-6)))   # calibrado
    true = np.where(correct, p.argmax(axis=1), (p.argmax(axis=1) + 1) % c)
    y = np.eye(c)[true]
    s = calibration_curve_stats(p, y, "multiclass")
    assert s["n"] == n
    assert s["slope"] == pytest.approx(1.0, abs=4 * s["slope_se"])
    assert len(s["per_class"]) == c


def test_multilabel_pools_elements_and_reports_per_class():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, (8000, 3))
    y = (rng.random(p.shape) < _sigmoid(0.4 * _logit(p))).astype(float)   # sobreconfiado
    s = calibration_curve_stats(p, y, "multilabel")
    assert s["n"] == p.size and s["slope"] == pytest.approx(0.4, abs=0.06)
    assert len(s["per_class"]) == 3
    assert all(c["n"] == 8000 and c["slope"] == pytest.approx(0.4, abs=0.2)
               for c in s["per_class"])


def test_undefined_cases_give_nan_not_exceptions():
    p = np.linspace(0.1, 0.9, 50)[:, None]
    all_one = calibration_curve_stats(p, np.ones((50, 1)), "binary")
    assert np.isnan(all_one["slope"]) and not all_one["converged"]
    const = calibration_curve_stats(np.full((50, 1), 0.3), (np.arange(50) % 2)[:, None], "binary")
    assert np.isnan(const["slope"]) and np.isfinite(const["intercept_citl"])
    sep = calibration_curve_stats(p, (p > 0.5).astype(float), "binary")          # separación
    assert np.isnan(sep["slope"]) and not sep["converged"]
    empty = calibration_curve_stats(np.empty((0, 2)), np.empty((0, 2)), "multilabel")
    assert np.isnan(empty["slope"]) and empty["n"] == 0


def test_saturated_float16_probabilities_do_not_break_the_fit():
    p, y = _simulate(0.0, 1.0, n=5000, seed=6)
    p[:50] = 1.0
    p[50:100] = 0.0
    out = calibration_curve_stats(p.astype(np.float16), y, "binary")
    assert np.isfinite(out["slope"])


# ------------------------------------------------------------------ ECE por clase

def test_per_class_ece_known_value():
    # Clase 0: siempre p = 0,9 pero positiva la mitad de las veces -> ECE 0,4.
    p = np.column_stack([np.full(10, 0.9), np.full(10, 0.1)])
    y = np.column_stack([[1] * 5 + [0] * 5, [0] * 10])
    out = per_class_calibration(p, y, "multilabel")
    assert out["ece"][0] == pytest.approx(0.4) and out["mce"][0] == pytest.approx(0.4)
    assert out["ece"][1] == pytest.approx(0.1)           # p = 0,1 y ningún positivo
    assert out["sce"] == pytest.approx(0.25)
    assert out["n_pos"].tolist() == [5, 0] and out["n"] == 10
    assert out["brier"][0] == pytest.approx(0.5 * 0.01 + 0.5 * 0.81)
    assert out["count"][0].sum() == 10 and out["edges"].shape == (16,)
    b = int(0.9 * 15)
    assert out["confidence"][0][b] == pytest.approx(0.9)
    assert out["frequency"][0][b] == pytest.approx(0.5)


def test_per_class_ece_is_zero_when_calibrated_and_weights_bins():
    # Dos grupos: p = 0,8 con 80 % positivos (8 de 10) y p = 0,3 con 30 % (3 de 10).
    p = np.array([0.8] * 10 + [0.3] * 10)[:, None]
    y = np.array([1] * 8 + [0] * 2 + [1] * 3 + [0] * 7, dtype=float)[:, None]
    assert per_class_calibration(p, y, "binary")["ece"][0] == pytest.approx(0.0, abs=1e-12)
    # Pesos por bin: 10 con hueco 0,2 y 10 con hueco 0,3 -> media 0,25
    y2 = np.array([1] * 6 + [0] * 4 + [1] * 6 + [0] * 4, dtype=float)[:, None]
    assert per_class_calibration(p, y2, "binary")["ece"][0] == pytest.approx(0.25)


def test_per_class_multiclass_is_one_vs_rest_and_probability_one_goes_to_last_bin():
    p = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([[1, 0], [0, 1], [1, 0]])
    out = per_class_calibration(p, y, "multiclass", bins=10)
    assert out["ece"].tolist() == [0.0, 0.0]
    assert out["count"][0][-1] == 2 and out["count"][0][0] == 1


def test_per_class_handles_empty_and_bad_bins():
    out = per_class_calibration(np.empty((0, 2)), np.empty((0, 2)), "multilabel")
    assert np.isnan(out["ece"]).all() and np.isnan(out["sce"]) and out["n"] == 0
    with pytest.raises(ValueError):
        per_class_calibration(np.zeros((2, 2)), np.zeros((2, 2)), "multilabel", bins=0)


# ------------------------------------------------------------------ ECE adaptativo

def test_adaptive_bins_have_equal_mass():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), 1003)
    y = np.eye(4)[rng.integers(0, 4, 1003)]
    out = adaptive_ece(p, y, "multiclass", bins=10)
    counts = [b["count"] for b in out["bins"]]
    assert sum(counts) == 1003 and max(counts) - min(counts) <= 1 and len(counts) == 10
    assert all(b["lower"] <= b["confidence"] <= b["upper"] for b in out["bins"])
    ups = [b["upper"] for b in out["bins"]]
    assert ups == sorted(ups)


def test_adaptive_ece_known_value():
    # 4 predicciones, 2 bins: (0,6; 0,6) con aciertos (1, 0) y (0,9; 0,9) con (1, 1)
    p = np.array([[0.6, 0.4], [0.6, 0.4], [0.9, 0.1], [0.9, 0.1]])
    y = np.array([[1, 0], [0, 1], [1, 0], [1, 0]])
    out = adaptive_ece(p, y, "multiclass", bins=2)
    # bin 1: acc 0,5, conf 0,6 -> 0,1 ; bin 2: acc 1, conf 0,9 -> 0,1 ; ECE = 0,1 ; MCE = 0,1
    assert out["ece"] == pytest.approx(0.1) and out["mce"] == pytest.approx(0.1)


def test_adaptive_ece_equals_ece_when_one_bin_and_differs_otherwise():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3) * 0.3, 2000)
    y = np.eye(3)[rng.integers(0, 3, 2000)]
    one = adaptive_ece(p, y, "multiclass", bins=1)
    assert one["ece"] == pytest.approx(calibration(p, y, "multiclass", bins=1)["ece"])
    assert adaptive_ece(p, y, "multiclass", bins=15)["ece"] != \
        pytest.approx(calibration(p, y, "multiclass", bins=15)["ece"])


def test_adaptive_ece_is_small_for_calibrated_and_large_for_overconfident():
    rng = np.random.default_rng(2)
    n = 20000
    conf = rng.uniform(0.5, 1.0, n)
    good = rng.random(n) < conf
    bad = rng.random(n) < 0.5
    p = np.column_stack([conf, 1 - conf])
    for ok, lo, hi in ((good, 0, 0.03), (bad, 0.2, 0.3)):
        y = np.eye(2)[np.where(ok, 0, 1)]
        assert lo <= adaptive_ece(p, y, "multiclass", bins=15)["ece"] <= hi


def test_adaptive_binary_uses_confidence_of_the_predicted_label():
    p = np.array([[0.9], [0.2], [0.7], [0.4]])
    y = np.array([[1], [0], [0], [1]])
    out = adaptive_ece(p, y, "binary", bins=2)
    assert out["n"] == 4
    # confianzas 0,9 / 0,8 / 0,7 / 0,6 con aciertos 1 / 1 / 0 / 0
    assert out["ece"] == pytest.approx(0.5 * abs(0.0 - 0.65) + 0.5 * abs(1.0 - 0.85))


def test_adaptive_fewer_predictions_than_bins_and_classwise():
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    y = np.array([[1, 0], [0, 1], [0, 1]])
    out = adaptive_ece(p, y, "multiclass", bins=15)
    assert len(out["bins"]) == 3 and all(b["count"] == 1 for b in out["bins"])
    cw = adaptive_ece(p, y, "multiclass", bins=15, classwise=True)
    # un bin por muestra: |y - p| medio por clase
    assert cw["per_class"][0] == pytest.approx((0.1 + 0.2 + 0.6) / 3)
    assert cw["ece"] == pytest.approx(cw["per_class"].mean()) and cw["bins"] == []


def test_adaptive_empty_and_invalid():
    out = adaptive_ece(np.empty((0, 2)), np.empty((0, 2)), "multilabel")
    assert np.isnan(out["ece"]) and out["n"] == 0
    assert np.isnan(adaptive_ece(np.empty((0, 2)), np.empty((0, 2)), "multilabel",
                                 classwise=True)["ece"])
    with pytest.raises(ValueError):
        adaptive_ece(np.zeros((2, 2)), np.zeros((2, 2)), "multilabel", bins=0)
