"""Punto de operación: umbral fijo, sensibilidad objetivo, Youden y VPP/VPN."""
import numpy as np
import pytest

from src.analysis.confusion import confusion
from src.analysis.operating_point import (operating_point, operating_points, predictive_values,
                                          threshold_for_sensitivity, youden_threshold)
from src.analysis.thresholds import threshold_sweep

# Ejemplo clásico: ROC (0,0) (0,.5) (.5,.5) (.5,1) (1,1) con umbrales inf, .8, .4, .35, .1
SCORES = np.array([0.1, 0.4, 0.35, 0.8])[:, None]
TARGET = np.array([0, 0, 1, 1])[:, None]


def _random(task, n=400, c=3, seed=0):
    rng = np.random.default_rng(seed)
    if task == "multiclass":
        return rng.dirichlet(np.ones(c), n), np.eye(c)[rng.integers(0, c, n)]
    y = (rng.random((n, c)) < 0.3).astype(float)
    return np.clip(0.35 * y + 0.65 * rng.random((n, c)), 0, 1), y


# ------------------------------------------------------------------ valores predictivos

def test_predictive_values_by_bayes_known_numbers():
    ppv, npv = predictive_values(0.9, 0.8, 0.1)
    assert ppv == pytest.approx(0.09 / (0.09 + 0.18))
    assert npv == pytest.approx(0.72 / (0.72 + 0.01))
    ppv, npv = predictive_values([0.9, 0.9], [0.8, 0.8], [0.1, 0.5])
    assert ppv[1] == pytest.approx(0.45 / (0.45 + 0.1)) and ppv[0] < ppv[1]
    assert np.isnan(predictive_values(0.0, 1.0, 0.1)[0])         # ningún positivo predicho


# ------------------------------------------------------------------ umbral fijo

def test_fixed_threshold_known_counts():
    out = operating_point(SCORES, TARGET, "binary", threshold=0.5)
    assert (out["tp"][0], out["fp"][0], out["fn"][0], out["tn"][0]) == (1, 0, 1, 2)
    assert out["sensitivity"][0] == pytest.approx(0.5)
    assert out["specificity"][0] == pytest.approx(1.0)
    assert out["ppv"][0] == pytest.approx(1.0) and out["npv"][0] == pytest.approx(2 / 3)
    assert out["prevalence"][0] == pytest.approx(0.5) and out["threshold"] == 0.5


@pytest.mark.parametrize("task", ["multilabel", "binary", "multiclass"])
def test_observed_prevalence_gives_the_plain_precision_and_npv(task):
    p, y = _random(task, c=1 if task == "binary" else 3)
    out = operating_point(p, y, task, threshold=0.4)
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(out["ppv"], out["tp"] / (out["tp"] + out["fp"]))
        np.testing.assert_allclose(out["npv"], out["tn"] / (out["tn"] + out["fn"]))
    assert out["tp"].sum() + out["fn"].sum() == y.sum()


def test_matches_threshold_sweep_and_confusion():
    p, y = _random("multilabel")
    out = operating_point(p, y, "multilabel", 0.45)
    sweep = threshold_sweep(p, y, "multilabel", [0.45])
    np.testing.assert_allclose(out["sensitivity"], sweep["per_class"]["recall"][0], atol=1e-6)
    np.testing.assert_allclose(out["specificity"], sweep["per_class"]["specificity"][0], atol=1e-6)
    cm = confusion(p, y, "multilabel", 0.45)["per_class"]
    assert [c["tp"] for c in cm] == out["tp"].tolist()
    assert [c["fp"] for c in cm] == out["fp"].tolist()


def test_other_prevalence_changes_only_ppv_npv():
    p, y = _random("binary", c=1)
    base = operating_point(p, y, "binary")
    low = operating_point(p, y, "binary", prevalence=0.01)
    assert low["sensitivity"][0] == base["sensitivity"][0]
    assert low["ppv"][0] < base["ppv"][0] and low["npv"][0] > base["npv"][0]
    assert low["prevalence"][0] == 0.01
    per_class = operating_point(*_random("multilabel"), "multilabel", prevalence=[0.1, 0.2, 0.3])
    assert per_class["prevalence"].tolist() == [0.1, 0.2, 0.3]
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            operating_point(p, y, "binary", prevalence=bad)


def test_undefined_ratios_are_nan_and_multiclass_ignores_threshold():
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    y = np.array([[1, 0], [1, 0]])                      # la clase 1 no tiene positivos
    out = operating_point(p, y, "multiclass", threshold=0.99)
    assert np.isnan(out["sensitivity"][1]) and np.isnan(out["threshold"])
    assert out["sensitivity"][0] == pytest.approx(0.5)
    assert out["macro"]["sensitivity"] == pytest.approx(0.5)


# ------------------------------------------------------------------ sensibilidad objetivo

def test_sensitivity_target_known_values():
    half = threshold_for_sensitivity(SCORES, TARGET, "binary", target=0.5)
    assert half["score"][0] == pytest.approx(0.8)
    assert half["threshold"][0] == pytest.approx(0.6)            # entre 0,8 y 0,4
    assert half["sensitivity"][0] == pytest.approx(0.5) and half["specificity"][0] == 1.0
    full = threshold_for_sensitivity(SCORES, TARGET, "binary", target=0.9)
    assert full["score"][0] == pytest.approx(0.35)
    assert full["threshold"][0] == pytest.approx(0.225)          # entre 0,35 y 0,1
    assert full["sensitivity"][0] == 1.0 and full["specificity"][0] == pytest.approx(0.5)
    assert full["target"] == 0.9
    assert full["ppv"][0] == pytest.approx(2 / 3) and full["npv"][0] == pytest.approx(1.0)


@pytest.mark.parametrize("op", [lambda p, t: p > t, lambda p, t: p >= t])
def test_threshold_reproduces_the_reported_sensitivity_with_both_operators(op):
    p, y = _random("multilabel", n=600)
    # empates fuertes, como float16
    p = np.round(p, 2)
    res = threshold_for_sensitivity(p, y, "multilabel", target=0.9)
    for c in range(3):
        pred = op(p[:, c], res["threshold"][c])
        sens = (pred & (y[:, c] > 0)).sum() / (y[:, c] > 0).sum()
        spec = (~pred & (y[:, c] == 0)).sum() / (y[:, c] == 0).sum()
        assert sens == pytest.approx(res["sensitivity"][c])
        assert spec == pytest.approx(res["specificity"][c])
        assert sens >= 0.9


def test_highest_threshold_meeting_the_target_is_optimal_by_brute_force():
    p, y = _random("binary", c=1, n=300, seed=3)
    res = threshold_for_sensitivity(p, y, "binary", target=0.95)
    pos = y[:, 0] > 0
    best_spec = max(((p[:, 0] < t) & ~pos).sum() / (~pos).sum()
                    for t in np.unique(p[:, 0]) if (p[pos, 0] >= t).mean() >= 0.95)
    assert res["specificity"][0] == pytest.approx(best_spec)


def test_higher_target_never_raises_specificity():
    p, y = _random("multilabel", n=800, seed=4)
    lo = threshold_for_sensitivity(p, y, "multilabel", 0.80)
    hi = threshold_for_sensitivity(p, y, "multilabel", 0.95)
    assert np.all(hi["specificity"] <= lo["specificity"] + 1e-12)
    assert np.all(hi["threshold"] <= lo["threshold"])


def test_sensitivity_one_and_invalid_targets():
    res = threshold_for_sensitivity(SCORES, TARGET, "binary", target=1.0)
    assert res["sensitivity"][0] == 1.0
    for bad in (0.0, 1.5, -1):
        with pytest.raises(ValueError):
            threshold_for_sensitivity(SCORES, TARGET, "binary", target=bad)


def test_class_without_positives_or_negatives_is_nan():
    p, y = _random("multilabel")
    y[:, 0] = 0
    y[:, 1] = 1
    res = threshold_for_sensitivity(p, y, "multilabel")
    assert np.isnan(res["threshold"][:2]).all() and np.isfinite(res["threshold"][2])
    assert res["macro"]["specificity"] == pytest.approx(res["specificity"][2])
    yj = youden_threshold(p, y, "multilabel")
    assert np.isnan(yj["j"][:2]).all()


def test_lowest_score_cut_is_below_every_score():
    p = np.array([[0.2], [0.5], [0.9]])
    y = np.array([[1], [1], [0]])
    res = threshold_for_sensitivity(p, y, "binary", target=1.0)
    assert res["score"][0] == pytest.approx(0.2)
    assert res["threshold"][0] < 0.2 and (p[:, 0] > res["threshold"][0]).all()


# ------------------------------------------------------------------ Youden

def test_youden_known_value():
    res = youden_threshold(SCORES, TARGET, "binary")
    assert res["j"][0] == pytest.approx(0.5)
    assert res["score"][0] == pytest.approx(0.8)                 # primer máximo: umbral más alto
    assert res["threshold"][0] == pytest.approx(0.6)
    assert res["sensitivity"][0] == pytest.approx(0.5) and res["specificity"][0] == 1.0


def test_youden_is_the_brute_force_maximum():
    p, y = _random("multilabel", n=500, seed=5)
    res = youden_threshold(p, y, "multilabel")
    for c in range(3):
        pos = y[:, c] > 0
        js = [(p[pos, c] >= t).mean() + (p[~pos, c] < t).mean() - 1 for t in np.unique(p[:, c])]
        assert res["j"][c] == pytest.approx(max(js))
        pred = p[:, c] > res["threshold"][c]
        assert pred[pos].mean() + (~pred[~pos]).mean() - 1 == pytest.approx(res["j"][c])


def test_youden_without_discrimination_still_returns_a_real_threshold():
    p = np.array([[0.1], [0.2], [0.3], [0.4]])
    y = np.array([[1], [1], [0], [0]])                           # al revés: J < 0 en todo punto
    res = youden_threshold(p, y, "binary")
    assert np.isfinite(res["threshold"][0]) and res["j"][0] <= 0


def test_multiclass_is_one_vs_rest_on_each_class_probability():
    p, y = _random("multiclass", n=500, seed=6)
    res = threshold_for_sensitivity(p, y, "multiclass", 0.9)
    assert res["threshold"].shape == (3,) and (res["sensitivity"] >= 0.9).all()


# ------------------------------------------------------------------ paquete

def test_operating_points_bundle():
    p, y = _random("multilabel")
    out = operating_points(p, y, "multilabel", threshold=0.4, sensitivities=(0.9, 0.95))
    assert out["fixed"]["threshold"] == 0.4
    assert [r["target"] for r in out["sensitivity"]] == [0.9, 0.95]
    assert out["youden"]["j"].shape == (3,)
    assert out["sensitivity"][0]["prevalence"].tolist() == out["fixed"]["prevalence"].tolist()
