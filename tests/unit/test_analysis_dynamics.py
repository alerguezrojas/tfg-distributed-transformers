"""dynamics: cartografía de datos, prob_of_true, flips y peores errores."""
import numpy as np
import pytest

from src.analysis.dynamics import (cartography, flips, prob_of_true, sample_correct,
                                   worst_errors)


def test_prob_of_true_multiclass():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    y = np.array([[1, 0, 0], [0, 1, 0]])
    assert prob_of_true(p, y, "multiclass").tolist() == pytest.approx([0.7, 0.3])


def test_prob_of_true_multilabel_uses_p_for_positives_and_1_minus_p_for_negatives():
    p = np.array([[0.9, 0.2, 0.6]])
    y = np.array([[1, 0, 0]])
    assert prob_of_true(p, y, "multilabel")[0] == pytest.approx((0.9 + 0.8 + 0.4) / 3)
    assert prob_of_true(np.array([[0.3]]), np.array([[0]]), "binary")[0] == pytest.approx(0.7)


def test_sample_correct_conventions():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    y = np.array([[1, 0, 0], [0, 1, 0]])
    assert sample_correct(p, y, "multiclass").tolist() == [True, False]
    pm = np.array([[0.9, 0.1], [0.6, 0.1], [0.9, 0.9], [0.1, 0.1]])
    ym = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
    assert sample_correct(pm, ym, "multilabel").tolist() == [True, True, False, False]  # exact
    assert sample_correct(pm, ym, "multilabel", threshold=0.7).tolist() == [True, False, False, False]


def test_cartography_statistics_are_the_paper_definitions():
    pt = np.array([[0.9, 0.1], [0.7, 0.5], [0.8, 0.9]])      # E=3, M=2
    ok = np.array([[1, 0], [1, 1], [1, 1]])
    out = cartography(pt, ok)
    assert out["confidence"].tolist() == pytest.approx(pt.mean(axis=0).tolist())
    assert out["variability"].tolist() == pytest.approx(pt.std(axis=0, ddof=0).tolist())
    assert out["correctness"].tolist() == pytest.approx([1.0, 2 / 3])


def test_cartography_regions():
    e = 6
    # Muestra 0: siempre alta (fácil). 1: siempre baja (difícil). 2: oscila (ambigua).
    # Muestras 3-8: fáciles con variabilidad ínfima (para que el tercio superior sea la 2).
    easy = np.full(e, 0.95)
    hard = np.full(e, 0.05)
    amb = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    rest = [0.9 + 0.001 * np.arange(e) * k for k in range(1, 7)]
    pt = np.stack([easy, hard, amb, *rest], axis=1)
    correct = pt > 0.5
    out = cartography(pt, correct)
    assert out["region"][0] == "easy"
    assert out["region"][1] == "hard"
    assert out["region"][2] == "ambiguous"
    assert set(out["region"].tolist()) <= {"easy", "hard", "ambiguous"}
    assert out["thresholds"]["confidence"] == 0.5
    assert out["variability"][2] > out["thresholds"]["variability"]
    # Una tercera parte de 9 muestras como máximo es ambigua
    assert (out["region"] == "ambiguous").sum() <= 3
    assert out["correctness"][2] == pytest.approx(0.5)


def test_cartography_single_epoch_has_no_ambiguous():
    pt = np.array([[0.9, 0.1, 0.5, 0.2]])
    out = cartography(pt, pt > 0.5)
    assert out["variability"].tolist() == [0.0] * 4
    assert out["region"].tolist() == ["easy", "hard", "easy", "hard"]


def test_cartography_custom_thresholds_and_validation():
    pt = np.array([[0.9, 0.6], [0.9, 0.6]])
    out = cartography(pt, pt > 0.5, confidence_split=0.8)
    assert out["region"].tolist() == ["easy", "hard"]
    with pytest.raises(ValueError):
        cartography(np.zeros((3, 4)), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        cartography(np.zeros(4), np.zeros(4))


def test_flips_both_directions():
    y = np.eye(2, dtype=np.uint8)[[0, 0, 1, 1]]
    prev = np.array([[.9, .1], [.9, .1], [.2, .8], [.9, .1]])   # ok ok ok mal
    curr = np.array([[.9, .1], [.2, .8], [.2, .8], [.1, .9]])   # ok mal ok ok
    out = flips(prev, curr, y, "multiclass")
    assert out["to_wrong"].tolist() == [1]
    assert out["to_right"].tolist() == [3]


def test_flips_multilabel_uses_threshold():
    y = np.array([[1, 0], [1, 0]])
    prev = np.array([[0.9, 0.1], [0.4, 0.1]])
    curr = np.array([[0.9, 0.1], [0.45, 0.1]])
    assert flips(prev, curr, y, "multilabel", 0.5)["to_right"].size == 0
    assert flips(prev, curr, y, "multilabel", 0.42)["to_right"].tolist() == [1]


def test_worst_errors_orders_by_loss():
    p = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.6, 0.4]])
    y = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
    loss = np.array([0.1, 0.9, 2.5, 0.5])
    out = worst_errors(p, y, loss, "multiclass", 2)
    assert out["indices"].tolist() == [2, 1]
    assert out["loss"].tolist() == pytest.approx([2.5, 0.9])
    assert out["targets"].tolist() == [[1, 0], [1, 0]]
    assert out["preds"].tolist() == [[0, 1], [0, 1]]
    assert out["preds"].dtype == np.uint8


def test_worst_errors_k_larger_than_n_nan_last_and_multilabel_preds():
    p = np.array([[0.9, 0.6], [0.1, 0.2]])
    y = np.array([[1, 0], [0, 1]])
    out = worst_errors(p, y, np.array([np.nan, 1.0]), "multilabel", 10)
    assert out["indices"].tolist() == [1, 0]
    assert worst_errors(p, y, np.array([1.0, 2.0]), "multilabel", 1)["preds"].tolist() == [[0, 0]]
    assert worst_errors(p, y, np.array([1.0, 2.0]), "multilabel", 0)["indices"].size == 0
    with pytest.raises(ValueError):
        worst_errors(p, y, np.array([1.0]), "multilabel", 1)
