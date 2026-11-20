"""confusion: conteos, índices por celda y TP/FP/FN/TN por clase."""
import numpy as np
import pytest

from src.analysis.confusion import confusion


def test_multiclass_matrix_and_cells():
    y = np.eye(3, dtype=np.uint8)[[0, 0, 1, 1, 2, 2]]
    pred_cls = [0, 1, 1, 1, 0, 2]
    p = np.eye(3)[pred_cls] * 0.8 + 0.1
    out = confusion(p, y, "multiclass")
    assert out["matrix"].tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert out["matrix"].sum() == 6
    assert out["cells"][(0, 1)].tolist() == [1]
    assert out["cells"][(1, 1)].tolist() == [2, 3]
    assert out["cells"][(2, 0)].tolist() == [4]
    assert out["cells"][(1, 2)].size == 0
    for (i, j), idx in out["cells"].items():
        assert len(idx) == out["matrix"][i, j]
    cls1 = out["per_class"][1]
    assert (cls1["tp"], cls1["fp"], cls1["fn"], cls1["tn"]) == (2, 1, 0, 3)
    assert cls1["fp_idx"].tolist() == [1]


def test_multilabel_is_co_activation_in_counts():
    y = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 0]])
    p = np.array([[.9, .2, .7], [.8, .1, .1], [.2, .9, .9], [.6, .1, .1]])
    out = confusion(p, y, "multilabel", threshold=0.5)
    pred = p > 0.5
    # celda (i, j) = nº de muestras con etiqueta i y predicción j
    assert out["matrix"].tolist() == (y.T @ pred.astype(int)).tolist()
    assert out["matrix"][0, 0] == 2 and out["matrix"][0, 2] == 1
    assert out["cells"][(0, 2)].tolist() == [0]
    assert out["cells"][(1, 1)].tolist() == [2]
    c0 = out["per_class"][0]
    assert (c0["tp"], c0["fp"], c0["fn"], c0["tn"]) == (2, 1, 0, 1)
    assert c0["fp_idx"].tolist() == [3] and c0["tn_idx"].tolist() == [2]
    for c in out["per_class"]:
        assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == 4
        assert c["tp_idx"].size == c["tp"] and c["fn_idx"].size == c["fn"]


def test_multilabel_matches_confusion_decorator_convention():
    """Normalizada por filas, es la matriz que escribe ConfusionMatrixDecorator."""
    rng = np.random.default_rng(0)
    y = (rng.random((200, 5)) < 0.3).astype(np.uint8)
    p = rng.random((200, 5))
    out = confusion(p, y, "multilabel", threshold=0.4)
    pred = (p > 0.4).astype(float)
    co = y.T.astype(float) @ pred
    rows = np.where(y.sum(axis=0) == 0, 1, y.sum(axis=0))
    expected = co / rows[:, None]                           # misma fórmula que el decorador
    assert np.allclose(out["matrix"] / rows[:, None], expected)


def test_threshold_changes_multilabel_predictions_only():
    y = np.array([[1], [0]])
    p = np.array([[0.4], [0.3]])
    assert confusion(p, y, "binary", threshold=0.5)["per_class"][0]["tp"] == 0
    assert confusion(p, y, "binary", threshold=0.35)["per_class"][0]["tp"] == 1
    y3 = np.eye(3, dtype=np.uint8)[[0, 1]]
    p3 = np.array([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]])
    a = confusion(p3, y3, "multiclass", threshold=0.99)["matrix"]
    assert a.tolist() == confusion(p3, y3, "multiclass", threshold=0.01)["matrix"].tolist()


def test_binary_task_single_column():
    y = np.array([[1], [1], [0], [0]])
    p = np.array([[0.9], [0.3], [0.8], [0.1]])
    out = confusion(p, y, "binary")
    assert out["matrix"].shape == (1, 1) and out["matrix"][0, 0] == 1
    c = out["per_class"][0]
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
    assert c["fn_idx"].tolist() == [1]


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confusion(np.zeros((3, 2)), np.zeros((3, 3)), "multilabel")
