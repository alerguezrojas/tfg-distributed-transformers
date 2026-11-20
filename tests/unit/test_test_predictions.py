"""Predicciones de test en el Inspector: dónde se guardan (`test_preds_<ts>.npz`), cómo se
descubren y se leen desde la web (solo dentro de `logs/`) y cómo alimentan las vistas."""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.analysis.clinical import clinical_cutoffs
from src.analysis.io import (
    default_predictions_path, list_test_predictions, run_dir_of_checkpoint, run_stamp_for,
    save_predictions,
)
from src.web import inspector_data as d
from src.web.inspector_stats import cutoffs_frame
from tests.fixtures.make_inspector_run import make_inspector_run

CLASSES = ["g0", "g1", "g2"]
PROBS = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .2, .7], [.1, .6, .3]])
TARGETS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])


def _save(path, **kw):
    return save_predictions(path, PROBS, TARGETS, 0.4, CLASSES, "multiclass", **kw)


def _when(stamp: str) -> float:
    return datetime.strptime(stamp, "%d%m%Y_%H%M%S").timestamp()


# ── Convención de ruta ──────────────────────────────────────────────────────────

def test_run_dir_of_checkpoint_maps_checkpoints_to_logs():
    assert run_dir_of_checkpoint("checkpoints/local/panda/single/vit/checkpoint_epoch_003.pt") \
        == Path("logs/local/panda/single/vit")
    assert run_dir_of_checkpoint("/data/checkpoints/verode/ddp/m/c.pt") \
        == Path("/data/logs/verode/ddp/m")
    assert run_dir_of_checkpoint("somewhere/else/c.pt") is None


def test_run_stamp_picks_the_run_that_wrote_the_checkpoint(tmp_path):
    for ts in ("01112026_100000", "05112026_100000", "18112026_090000"):
        (tmp_path / f"train_{ts}.log").write_text("")
    (tmp_path / "train_legacy.log").write_text("")
    assert run_stamp_for(tmp_path, _when("06112026_120000")) == "05112026_100000"
    assert run_stamp_for(tmp_path, _when("20112026_120000")) == "18112026_090000"
    assert run_stamp_for(tmp_path, _when("01102026_000000")) == "18112026_090000"   # sin anterior
    assert run_stamp_for(tmp_path) == "18112026_090000"
    assert run_stamp_for(tmp_path / "missing") is None


def test_default_predictions_path_uses_the_run_stamp(tmp_path):
    logs = tmp_path / "logs" / "local" / "single" / "vit"
    logs.mkdir(parents=True)
    (logs / "train_05112026_100000.log").write_text("")
    ckpt = tmp_path / "checkpoints" / "local" / "single" / "vit" / "checkpoint_epoch_001.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    os.utime(ckpt, (_when("06112026_120000"),) * 2)
    path = default_predictions_path(ckpt)
    assert path == logs / "test_preds_05112026_100000.npz"


def test_default_predictions_path_falls_back_to_evaluation_stamp(tmp_path):
    now = datetime(2026, 11, 20, 9, 30, 5)
    path = default_predictions_path(tmp_path / "ckpt.pt", now)        # fuera de checkpoints/
    assert path == Path("logs/test_preds_20112026_093005.npz")
    nolog = tmp_path / "checkpoints" / "local" / "single" / "vit" / "c.pt"   # sin train_*.log
    assert default_predictions_path(nolog, now).name == "test_preds_20112026_093005.npz"


# ── Descubrimiento y lectura segura ─────────────────────────────────────────────

def test_discovery_attaches_test_predictions(tmp_path):
    folder = make_inspector_run(tmp_path, ts="25102026_120000")
    _save(folder / "test_preds_25102026_120000.npz")
    only = make_inspector_run(tmp_path, ts="24102026_100000", with_events=False,
                              with_inspect=False, dataset="medmnist_derma")
    _save(only / "test_preds_23102026_100000.npz")                 # run sin events ni inspect
    (folder / "test_preds_bad.npz").write_bytes(b"")                # nombre sin sello: se ignora
    runs = {r.timestamp: r for r in d.discover_inspector_runs(tmp_path)}
    assert runs["25102026_120000"].test_preds == folder / "test_preds_25102026_120000.npz"
    assert runs["25102026_120000"].events_path and runs["25102026_120000"].inspect_dir
    assert runs["23102026_100000"].test_preds and runs["23102026_100000"].inspect_dir is None
    assert {s for s, _ in list_test_predictions(tmp_path)} == {"23102026_100000",
                                                              "25102026_120000"}


def test_load_test_predictions_only_inside_logs(tmp_path):
    inside = _save(tmp_path / "logs" / "local" / "test_preds_25102026_120000.npz")
    outside = _save(tmp_path / "other" / "test_preds_25102026_120000.npz")
    pred = d.load_test_predictions(inside, tmp_path)
    assert pred.probs.shape == (4, 3) and pred.task == "multiclass"
    with pytest.raises(ValueError, match="outside"):
        d.load_test_predictions(outside, tmp_path)
    link = tmp_path / "logs" / "link.npz"                            # un enlace no lo esquiva
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        d.load_test_predictions(link, tmp_path)


def test_load_test_predictions_refuses_pickled_objects(tmp_path):
    bad = tmp_path / "logs" / "test_preds_25102026_120000.npz"
    bad.parent.mkdir(parents=True)
    np.savez(bad, probs=np.array([{"a": 1}], dtype=object), targets=np.zeros((1, 1)),
             threshold=0.5, classes=np.array(["a"]), task=np.asarray("binary"))
    with pytest.raises(ValueError):
        d.load_test_predictions(bad, tmp_path)


# ── Del .npz a las vistas ───────────────────────────────────────────────────────

def test_predictions_as_epoch_feeds_the_monitor_views(tmp_path):
    path = _save(tmp_path / "logs" / "p.npz", ordinal=True,
                 groups={"centre": np.array(["a", "a", "b", "b"])})
    pred = d.load_test_predictions(path, tmp_path)
    monitor, epoch = d.predictions_as_epoch(pred)
    assert monitor.task == "multiclass" and monitor.classes == CLASSES
    assert monitor.targets.shape == (4, 3) and monitor.indices.tolist() == [0, 1, 2, 3]
    assert epoch.probs.shape == (4, 3) and epoch.threshold == pytest.approx(0.4)
    assert epoch.val_probs is None and not epoch.loss.any()


def test_inspector_ctx_from_test(tmp_path):
    from src.web.tabs.inspector import view_names
    from src.web.tabs.inspector._common import InspectorCtx
    path = _save(tmp_path / "logs" / "p.npz", ordinal=True,
                 groups={"centre": np.array(["a", "a", "b", "b"])})
    pred = d.load_test_predictions(path, tmp_path)
    c = InspectorCtx.from_test(pred, path)
    assert c.is_test and c.has_monitor and c.epoch == 0 and c.ordinal
    assert c.key == str(path) and c.epoch_mtime(0) > 0 and c.epoch_label == "test set"
    assert c.epoch_data().probs.shape == (4, 3) and list(c.groups) == ["centre"]
    assert view_names(c) == ["Curves", "Clinical", "Calibration", "Subgroups"]


def test_cutoffs_frame_matches_clinical_cutoffs():
    df = cutoffs_frame(clinical_cutoffs(PROBS, TARGETS, CLASSES))
    assert list(df["Cut-off"]) == [">= g1", ">= g2"]
    assert list(df.columns)[:3] == ["Cut-off", "n+", "n-"]
    assert df["Sensitivity"].tolist() == pytest.approx([1.0, 0.5])
    assert cutoffs_frame([]).empty
