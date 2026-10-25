"""io: lectura del directorio de inspección y comprobación del generador de fixtures."""
import numpy as np
import pytest

from src.analysis import (calibration, cartography, confusion, flips, list_epochs, load_epoch,
                          load_monitor_set, per_class_curves, prob_of_true, project_2d,
                          sample_correct, threshold_sweep, worst_errors)
from tests.fixtures.make_inspect_fixture import make_inspect_fixture


@pytest.mark.parametrize("task,c", [("multilabel", 5), ("multiclass", 4), ("binary", 1)])
def test_fixture_follows_the_contract(tmp_path, task, c):
    d = make_inspect_fixture(tmp_path, task=task, c=c, m=32, n_val=100, epochs=(1, 2, 3))
    assert d.name == "inspect_25102026_120000"

    ms = load_monitor_set(d)
    m = 32
    assert ms.indices.shape == (m,) and ms.indices.dtype == np.int64
    assert ms.targets.shape == (m, c) and ms.targets.dtype == np.uint8
    assert ms.thumbs.shape[0] == m and ms.thumbs.shape[-1] == 3 and ms.thumbs.dtype == np.uint8
    assert ms.thumbs.ndim == 4 and ms.thumbs.shape[1] == ms.thumbs.shape[2]
    assert ms.panel.dtype == np.int64 and ms.panel.max() < m
    assert len(ms.classes) == c and all(isinstance(k, str) for k in ms.classes)
    assert ms.task == task
    if task == "multiclass":
        assert (ms.targets.sum(axis=1) == 1).all()

    assert list_epochs(d) == [1, 2, 3]
    e = load_epoch(d, 2)
    assert e.epoch == 2
    assert e.probs.shape == (m, c) and e.probs.dtype == np.float16
    assert e.loss.shape == (m,) and e.loss.dtype == np.float32
    assert e.embeddings.shape[0] == m and e.embeddings.dtype == np.float16
    p = len(ms.panel)
    assert e.cam.shape[0] == p and e.cam.ndim == 3 and e.cam.dtype == np.uint8
    assert e.cam_class.shape == (p,) and e.cam_class.dtype == np.int64
    assert e.val_probs.shape == (100, c) and e.val_probs.dtype == np.float16
    assert e.val_targets.shape == (100, c) and e.val_targets.dtype == np.uint8
    assert e.threshold == pytest.approx(0.5)


@pytest.mark.parametrize("task,c", [("multilabel", 5), ("multiclass", 4), ("binary", 1)])
def test_every_analysis_function_runs_on_the_fixture(tmp_path, task, c):
    d = make_inspect_fixture(tmp_path, task=task, c=c, m=40, n_val=150)
    ms = load_monitor_set(d)
    ep = [load_epoch(d, k) for k in list_epochs(d)]
    y = ms.targets

    curves = per_class_curves(ep[-1].val_probs, ep[-1].val_targets)
    assert curves["macro"]["auc"] > 0.8                      # la señal mejora con la época
    assert 0 <= calibration(ep[-1].probs, y, task)["ece"] <= 1
    assert threshold_sweep(ep[-1].val_probs, ep[-1].val_targets, task)["f1"].size >= 1
    assert confusion(ep[-1].probs, y, task)["matrix"].shape == (c, c)

    pt = np.stack([prob_of_true(e.probs, y, task) for e in ep])
    ok = np.stack([sample_correct(e.probs, y, task) for e in ep])
    carto = cartography(pt, ok)
    assert carto["region"].shape == (40,)
    assert pt[-1].mean() > pt[0].mean()                      # mejora entre épocas

    assert set(flips(ep[0].probs, ep[-1].probs, y, task)) == {"to_wrong", "to_right"}
    assert len(worst_errors(ep[-1].probs, y, ep[-1].loss, task, 5)["indices"]) == 5
    assert project_2d(ep[-1].embeddings).shape == (40, 2)


def test_optional_keys_may_be_absent(tmp_path):
    d = make_inspect_fixture(tmp_path, m=16, epochs=(1,))
    full = dict(np.load(d / "epoch_001.npz"))
    for k in ("embeddings", "cam", "cam_class", "val_probs", "val_targets", "threshold"):
        full.pop(k)
    np.savez_compressed(d / "epoch_001.npz", **full)
    e = load_epoch(d, 1)
    assert e.probs.shape[0] == 16
    assert e.embeddings is None and e.cam is None and e.cam_class is None
    assert e.val_probs is None and e.val_targets is None and e.threshold is None

    ms = dict(np.load(d / "monitor_set.npz"))
    ms.pop("thumbs"), ms.pop("panel")
    np.savez_compressed(d / "monitor_set.npz", **ms)
    loaded = load_monitor_set(d)
    assert loaded.thumbs is None and loaded.panel is None and loaded.task == "multilabel"


def test_missing_required_keys_and_files(tmp_path):
    d = make_inspect_fixture(tmp_path, m=8, epochs=(1,))
    np.savez_compressed(d / "epoch_002.npz", probs=np.zeros((8, 5), dtype=np.float16))
    with pytest.raises(ValueError, match="loss"):
        load_epoch(d, 2)
    with pytest.raises(FileNotFoundError):
        load_epoch(d, 9)
    with pytest.raises(FileNotFoundError):
        load_monitor_set(tmp_path)


def test_list_epochs_sorted_and_tolerant(tmp_path):
    assert list_epochs(tmp_path / "no_existe") == []
    d = make_inspect_fixture(tmp_path, m=8, epochs=(10, 2, 1))
    (d / "notas.txt").write_text("x")
    (d / "epoch_abc.npz").write_bytes(b"")
    assert list_epochs(d) == [1, 2, 10]
    assert list_epochs(str(d)) == [1, 2, 10]


def test_fixture_is_reproducible_and_rejects_unknown_task(tmp_path):
    a = make_inspect_fixture(tmp_path / "a", m=12, epochs=(1,))
    b = make_inspect_fixture(tmp_path / "b", m=12, epochs=(1,))
    assert np.array_equal(load_epoch(a, 1).probs, load_epoch(b, 1).probs)
    with pytest.raises(ValueError):
        make_inspect_fixture(tmp_path, task="regression")


def test_analysis_package_does_not_import_torch():
    """src.analysis es puro: ni torch ni streamlit (se comprueba en un intérprete limpio)."""
    import subprocess
    import sys
    code = ("import sys, src.analysis; "
            "bad = [m for m in ('torch', 'streamlit') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
