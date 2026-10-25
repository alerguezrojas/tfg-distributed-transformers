"""calibration: ECE/MCE/Brier con casos conocidos en las tres tareas."""
import math

import numpy as np
import pytest

from src.analysis.calibration import calibration


def test_perfectly_calibrated_multiclass_has_zero_ece():
    # Confianza 0,8 con 8 de 10 aciertos, y 0,6 con 3 de 5.
    p = np.array([[0.8, 0.2]] * 10 + [[0.4, 0.6]] * 5)
    y = np.array([[1, 0]] * 8 + [[0, 1]] * 2 + [[0, 1]] * 3 + [[1, 0]] * 2)
    out = calibration(p, y, "multiclass")
    assert out["ece"] == pytest.approx(0.0, abs=1e-12)
    assert out["mce"] == pytest.approx(0.0, abs=1e-12)
    assert out["n"] == 15
    assert sum(b["count"] for b in out["bins"]) == 15


def test_overconfident_predictions_known_ece():
    # Siempre confianza 0,9 pero acierta la mitad: ECE = |0,5 - 0,9| = 0,4.
    p = np.array([[0.9, 0.1]] * 10)
    y = np.array([[1, 0]] * 5 + [[0, 1]] * 5)
    out = calibration(p, y, "multiclass")
    assert out["ece"] == pytest.approx(0.4)
    assert out["mce"] == pytest.approx(0.4)
    full = [b for b in out["bins"] if b["count"]]
    assert len(full) == 1
    assert full[0]["confidence"] == pytest.approx(0.9) and full[0]["accuracy"] == pytest.approx(0.5)


def test_ece_is_weighted_by_bin_size_and_mce_is_max():
    # 8 muestras con confianza 0,9 y 100 % de acierto (hueco 0,1);
    # 2 con confianza 0,55 y 0 % de acierto (hueco 0,55).
    p = np.array([[0.9, 0.1]] * 8 + [[0.55, 0.45]] * 2)
    y = np.array([[1, 0]] * 8 + [[0, 1]] * 2)
    out = calibration(p, y, "multiclass")
    assert out["ece"] == pytest.approx(0.8 * 0.1 + 0.2 * 0.55)
    assert out["mce"] == pytest.approx(0.55)


def test_brier_multiclass():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    y = np.array([[1, 0], [1, 0]])
    # muestra 0: 0 ; muestra 1: 0,25 + 0,25 = 0,5 -> media 0,25
    assert calibration(p, y, "multiclass")["brier"] == pytest.approx(0.25)


def test_binary_perfect_calibration_and_brier():
    p = np.array([[0.8]] * 5 + [[0.2]] * 5)
    y = np.array([[1]] * 4 + [[0]] + [[1]] + [[0]] * 4)   # 0,8 -> 80 % positivos; 0,2 -> 20 %
    out = calibration(p, y, "binary")
    assert out["ece"] == pytest.approx(0.0, abs=1e-12)
    expected = (4 * 0.04 + 0.64 + 0.64 + 4 * 0.04) / 10
    assert out["brier"] == pytest.approx(expected)


def test_multilabel_confidence_is_per_element_and_in_upper_half():
    p = np.array([[0.9, 0.1, 0.5], [0.3, 0.7, 0.2]])
    y = np.array([[1, 0, 1], [0, 1, 0]])
    out = calibration(p, y, "multilabel")
    assert out["n"] == 6
    # p=0,5 predice positivo (>= 0,5) con confianza 0,5; el resto 0,9 / 0,9 / 0,7 / 0,7 / 0,8
    assert all(b["count"] == 0 for b in out["bins"] if b["upper"] <= 0.5)
    assert out["brier"] == pytest.approx(np.mean((p - y) ** 2))
    # Todas las predicciones son correctas -> ECE = media(1 - confianza)
    conf = np.array([0.9, 0.9, 0.5, 0.7, 0.7, 0.8])
    assert out["ece"] == pytest.approx(np.mean(1 - conf))


def test_confidence_one_goes_to_last_bin():
    out = calibration(np.array([[1.0, 0.0]]), np.array([[1, 0]]), "multiclass", bins=10)
    assert out["bins"][-1]["count"] == 1 and out["ece"] == pytest.approx(0.0)


def test_bins_edges_and_empty_bins_are_nan():
    out = calibration(np.array([[0.9, 0.1]]), np.array([[1, 0]]), "multiclass", bins=5)
    assert out["edges"].tolist() == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    empty = out["bins"][0]
    assert empty["count"] == 0 and math.isnan(empty["confidence"]) and math.isnan(empty["accuracy"])
    assert out["bins"][4]["lower"] == pytest.approx(0.8)


def test_empty_input_gives_nan_not_exception():
    out = calibration(np.zeros((0, 3)), np.zeros((0, 3)), "multilabel")
    assert math.isnan(out["ece"]) and math.isnan(out["mce"]) and out["n"] == 0


def test_float16_and_invalid_arguments():
    p = np.array([[0.9, 0.1]], dtype=np.float16)
    assert calibration(p, np.array([[1, 0]], dtype=np.uint8), "multiclass")["n"] == 1
    with pytest.raises(ValueError):
        calibration(p, np.array([[1, 0]]), "multiclass", bins=0)
    with pytest.raises(ValueError):
        calibration(p, np.array([[1, 0, 0]]), "multiclass")
