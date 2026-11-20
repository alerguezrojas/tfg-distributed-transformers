"""threshold_sweep: valores a mano y coincidencia con src/training/metrics.py."""
import numpy as np
import pytest

from src.analysis.thresholds import DEFAULT_GRID, threshold_sweep


def test_hand_computed_single_class():
    p = np.array([[0.9], [0.8], [0.3], [0.2], [0.6]])
    y = np.array([[1], [0], [1], [0], [1]])
    out = threshold_sweep(p, y, "binary", grid=[0.5])
    # pred = 1,1,0,0,1  ->  tp=2 fp=1 fn=1 tn=1
    assert out["precision"][0] == pytest.approx(2 / 3)
    assert out["recall"][0] == pytest.approx(2 / 3)
    assert out["sensitivity"][0] == pytest.approx(2 / 3)
    assert out["specificity"][0] == pytest.approx(1 / 2)
    assert out["f1"][0] == pytest.approx(2 / 3)


def test_default_grid_and_shapes():
    rng = np.random.default_rng(0)
    out = threshold_sweep(rng.random((50, 4)), rng.integers(0, 2, (50, 4)), "multilabel")
    assert out["thresholds"].tolist() == DEFAULT_GRID.tolist()
    assert 0.5 in out["thresholds"].tolist()
    for k in ("f1", "precision", "recall", "sensitivity", "specificity"):
        assert out[k].shape == (19,)
        assert out["per_class"][k].shape == (19, 4)
    assert out["best"]["f1"] == out["f1"].max()
    assert out["best"]["threshold"] in out["thresholds"]


def test_recall_decreases_and_specificity_increases_with_threshold():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, (300, 3))
    p = np.clip(0.4 * y + 0.6 * rng.random((300, 3)), 0, 1)
    out = threshold_sweep(p, y, "multilabel")
    assert np.all(np.diff(out["recall"]) <= 1e-12)
    assert np.all(np.diff(out["specificity"]) >= -1e-12)


def test_custom_grid_and_float16_inputs():
    p = np.array([[0.9], [0.1]], dtype=np.float16)
    y = np.array([[1], [0]], dtype=np.uint8)
    out = threshold_sweep(p, y, "binary", grid=[0.2, 0.95])
    assert out["thresholds"].tolist() == [0.2, 0.95]
    assert out["recall"][0] == pytest.approx(1.0) and out["recall"][1] == pytest.approx(0.0)


@pytest.mark.parametrize("task_name", ["multilabel", "binary"])
def test_matches_training_metrics_multilabel(task_name):
    torch = pytest.importorskip("torch")
    from src.training import metrics as m
    from src.training.tasks import get_task
    rng = np.random.default_rng(2)
    c = 1 if task_name == "binary" else 6
    probs = rng.random((400, c)).astype(np.float32)
    y = (rng.random((400, c)) < 0.3).astype(np.uint8)
    task = get_task(task_name)
    out = threshold_sweep(probs, y, task_name)
    for i, t in enumerate(out["thresholds"]):
        preds = task.predict(torch.tensor(probs), float(t)).bool()
        hard = torch.tensor(y)
        assert out["f1"][i] == pytest.approx(m.f1_score(preds, hard), abs=1e-6)
        assert out["precision"][i] == pytest.approx(m.precision(preds, hard), abs=1e-6)
        assert out["recall"][i] == pytest.approx(m.recall(preds, hard), abs=1e-6)


def test_matches_training_metrics_multiclass():
    torch = pytest.importorskip("torch")
    from src.training import metrics as m
    from src.training.tasks import get_task
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(300, 5))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.eye(5, dtype=np.uint8)[rng.integers(0, 5, 300)]
    out = threshold_sweep(probs, y, "multiclass", grid=[0.1, 0.9])     # la rejilla se ignora
    assert out["f1"].shape == (1,) and np.isnan(out["thresholds"][0])
    preds = get_task("multiclass").predict(torch.tensor(probs)).bool()
    hard = torch.tensor(y)
    assert out["f1"][0] == pytest.approx(m.f1_score(preds, hard), abs=1e-6)
    assert out["precision"][0] == pytest.approx(m.precision(preds, hard), abs=1e-6)
    assert out["recall"][0] == pytest.approx(m.recall(preds, hard), abs=1e-6)
    assert np.isnan(out["best"]["threshold"]) and out["best"]["f1"] == out["f1"][0]


def test_accepts_task_object_and_rejects_unknown_task():
    pytest.importorskip("torch")
    from src.training.tasks import get_task
    p, y = np.array([[0.7], [0.2]]), np.array([[1], [0]])
    a = threshold_sweep(p, y, get_task("binary"), grid=[0.5])
    b = threshold_sweep(p, y, "binary", grid=[0.5])
    assert a["f1"].tolist() == b["f1"].tolist()
    with pytest.raises(ValueError):
        threshold_sweep(p, y, "regression")
