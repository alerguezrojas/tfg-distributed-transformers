"""Curvas ROC/PR de src/analysis/curves.py: casos conocidos, comparación con metrics.py y límites."""
import math

import numpy as np
import pytest

from src.analysis.curves import per_class_curves, pr_curve, roc_curve

# Ejemplo clásico de la documentación de scikit-learn (valores conocidos).
SK_SCORES = [0.1, 0.4, 0.35, 0.8]
SK_TARGET = [0, 0, 1, 1]


def _brute_auc(scores, target):
    """AUC por definición: P(score_pos > score_neg) + 0.5 P(empate)."""
    s, y = np.asarray(scores, float), np.asarray(target) > 0
    pos, neg = s[y], s[~y]
    diff = pos[:, None] - neg[None, :]
    return ((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg))


def _brute_ap(scores, target):
    """AP por definición: media de la precisión en la posición de cada positivo (sin empates)."""
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    y = (np.asarray(target) > 0)[order]
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return prec[y].mean()


def test_roc_known_values():
    fpr, tpr, thr, auc = roc_curve(SK_SCORES, SK_TARGET)
    assert fpr.tolist() == [0.0, 0.0, 0.5, 0.5, 1.0]
    assert tpr.tolist() == [0.0, 0.5, 0.5, 1.0, 1.0]
    assert thr.tolist() == [math.inf, 0.8, 0.4, 0.35, 0.1]
    assert auc == pytest.approx(0.75)


def test_pr_known_values():
    precision, recall, thr, ap = pr_curve(SK_SCORES, SK_TARGET)
    assert thr.tolist() == [0.8, 0.4, 0.35, 0.1]
    assert precision.tolist() == pytest.approx([1.0, 0.5, 2 / 3, 0.5])
    assert recall.tolist() == [0.5, 0.5, 1.0, 1.0]
    assert ap == pytest.approx(5 / 6)           # 0,5·1 + 0,5·(2/3)


def test_perfect_and_inverted_scores():
    y = [0, 0, 1, 1, 0, 1]
    s = [0.1, 0.2, 0.9, 0.8, 0.3, 0.7]
    assert roc_curve(s, y)[3] == pytest.approx(1.0)
    assert pr_curve(s, y)[3] == pytest.approx(1.0)
    assert roc_curve([1 - v for v in s], y)[3] == pytest.approx(0.0)


def test_constant_scores_are_chance_level():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    fpr, tpr, _, auc = roc_curve(np.full(8, 0.3), y)
    assert auc == pytest.approx(0.5)
    assert fpr.tolist() == [0.0, 1.0] and tpr.tolist() == [0.0, 1.0]
    assert pr_curve(np.full(8, 0.3), y)[3] == pytest.approx(0.5)      # prevalencia


def test_random_scores_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.integers(0, 2, 300)
        s = rng.random(300)
        assert roc_curve(s, y)[3] == pytest.approx(_brute_auc(s, y))
        assert pr_curve(s, y)[3] == pytest.approx(_brute_ap(s, y))


def test_ties_count_half_in_auc():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = np.round(rng.random(400), 1)                 # muchos empates
    assert roc_curve(s, y)[3] == pytest.approx(_brute_auc(s, y))


def test_auc_matches_training_metrics():
    torch = pytest.importorskip("torch")
    from src.training.metrics import roc_auc
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500), 2)
    expected = roc_auc(torch.tensor(s), torch.tensor(y))
    assert roc_curve(s, y)[3] == pytest.approx(expected)


def test_float16_inputs_are_accepted():
    s = np.array(SK_SCORES, dtype=np.float16)
    assert roc_curve(s, np.array(SK_TARGET, dtype=np.uint8))[3] == pytest.approx(0.75)


@pytest.mark.parametrize("target", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_degenerate_class_gives_nan_not_exception(target):
    for fn in (roc_curve, pr_curve):
        a, b, c, score = fn([0.1, 0.5, 0.4, 0.9], target)
        assert math.isnan(score)
        assert a.size == b.size == c.size == 0


def test_long_curves_are_thinned_keeping_ends_and_exact_score():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 20000)
    s = rng.random(20000)
    fpr, tpr, thr, auc = roc_curve(s, y)
    assert len(fpr) <= 1000 and len(fpr) == len(tpr) == len(thr)
    assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert auc == pytest.approx(_brute_auc(s[:3000], y[:3000]), abs=0.05)
    prec, rec, pthr, ap = pr_curve(s, y)
    assert len(prec) <= 1000 and rec[-1] == pytest.approx(1.0)
    # La puntuación sale de la curva completa, no de la reducida.
    full = roc_curve(s, y, max_points=None)
    assert len(full[0]) > 10000 and full[3] == auc
    assert pr_curve(s, y, max_points=None)[3] == ap


def test_short_curves_are_not_thinned():
    assert len(roc_curve(SK_SCORES, SK_TARGET, max_points=1000)[0]) == 5
    assert len(roc_curve(SK_SCORES, SK_TARGET, max_points=3)[0]) == 3


def test_per_class_curves_structure_and_macro_micro():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, (400, 3))
    p = np.clip(0.6 * y + 0.5 * rng.random((400, 3)), 0, 1)
    out = per_class_curves(p, y)
    assert len(out["classes"]) == 3 and out["auc"].shape == (3,)
    for c in range(3):
        assert out["auc"][c] == pytest.approx(_brute_auc(p[:, c], y[:, c]))
        assert out["classes"][c]["roc"]["auc"] == out["auc"][c]
        assert out["ap"][c] == pytest.approx(_brute_ap(p[:, c], y[:, c]))
    assert out["macro"]["auc"] == pytest.approx(out["auc"].mean())
    assert out["macro"]["ap"] == pytest.approx(out["ap"].mean())
    assert out["micro"]["roc"]["auc"] == pytest.approx(_brute_auc(p.ravel(), y.ravel()))
    assert out["micro"]["pr"]["ap"] == pytest.approx(_brute_ap(p.ravel(), y.ravel()))


def test_per_class_curves_undefined_class_is_nan_and_excluded_from_macro():
    y = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])   # clase 1 sin positivos
    y[:, 2] = [1, 0, 1, 1]                                          # clase 2 válida
    p = np.array([[.9, .5, .8], [.2, .5, .1], [.8, .5, .7], [.1, .5, .9]])
    out = per_class_curves(p, y)
    assert math.isnan(out["auc"][1]) and math.isnan(out["ap"][1])
    assert out["classes"][1]["roc"]["fpr"].size == 0
    assert out["macro"]["auc"] == pytest.approx(np.nanmean(out["auc"]))
    assert not math.isnan(out["macro"]["auc"])


def test_per_class_curves_all_undefined_macro_is_nan():
    out = per_class_curves(np.full((4, 2), 0.5), np.zeros((4, 2)))
    assert math.isnan(out["macro"]["auc"]) and math.isnan(out["micro"]["roc"]["auc"])


def test_per_class_curves_multiclass_one_hot():
    y = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    out = per_class_curves(y * 0.9 + 0.05, y)                 # perfecto
    assert out["auc"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert out["micro"]["roc"]["auc"] == pytest.approx(1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        roc_curve([0.1, 0.2], [0, 1, 1])
    with pytest.raises(ValueError):
        per_class_curves(np.zeros((3, 2)), np.zeros((3, 3)))
