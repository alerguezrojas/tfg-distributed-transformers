"""Predicciones de test en el Inspector: dónde se guardan (`test_preds_<ts>.npz`), cómo se
descubren y se leen desde la web (solo dentro de `logs/`) y cómo alimentan las vistas."""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.analysis.clinical import clinical_cutoffs
from src.analysis.io import (
    default_predictions_path, list_test_predictions, run_location, run_stamp_for,
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

def test_run_location_swaps_checkpoints_for_logs_when_no_config():
    assert run_location("checkpoints/local/panda/single/vit/checkpoint_epoch_003.pt") \
        == (Path("logs/local/panda/single/vit"), None)
    assert run_location("/data/checkpoints/verode/ddp/m/c.pt") == (Path("/data/logs/verode/ddp/m"),
                                                                    None)
    assert run_location("somewhere/else/c.pt") == (None, None)


def test_run_location_uses_the_stamp_folder_of_new_checkpoints():
    loc = run_location("checkpoints/local/panda/single/vit/20112026_101500/best.pt")
    assert loc == (Path("logs/local/panda/single/vit"), "20112026_101500")


def test_run_location_follows_the_config_not_the_text():
    # checkpoint.dir = checkpoints/vit_tiny, output.env = local (configs/models/vit_tiny.yaml)
    ckpt = "checkpoints/vit_tiny/single/vit_tiny/20112026_101500/last.pt"
    assert run_location(ckpt, "checkpoints/vit_tiny", "local") \
        == (Path("logs/local/single/vit_tiny"), "20112026_101500")
    # un checkpoint fuera de checkpoint.dir cae a la deducción por texto
    assert run_location("elsewhere/x.pt", "checkpoints/vit_tiny", "local") == (None, None)


def test_run_stamp_picks_the_run_that_wrote_the_checkpoint(tmp_path):
    for name in ("train_01112026_100000.log", "train_deep_05112026_100000.log",
                 "events_18112026_090000.jsonl", "train_legacy.log"):
        (tmp_path / name).write_text("")
    assert run_stamp_for(tmp_path, _when("06112026_120000")) == ("05112026_100000", True)
    assert run_stamp_for(tmp_path, _when("20112026_120000")) == ("18112026_090000", True)
    assert run_stamp_for(tmp_path, _when("01102026_000000")) == ("18112026_090000", True)
    assert run_stamp_for(tmp_path) == ("18112026_090000", True)
    assert run_stamp_for(tmp_path / "missing") == (None, False)


def test_run_stamp_single_run_is_not_ambiguous(tmp_path):
    (tmp_path / "train_05112026_100000.log").write_text("")
    assert run_stamp_for(tmp_path, _when("06112026_120000")) == ("05112026_100000", False)
    assert run_stamp_for(tmp_path, _when("01112026_120000")) == ("05112026_100000", True)


def test_default_predictions_path_old_shared_folder_warns_when_ambiguous(tmp_path):
    logs = tmp_path / "logs" / "local" / "single" / "vit"
    logs.mkdir(parents=True)
    for ts in ("05112026_100000", "06112026_100000"):
        (logs / f"train_{ts}.log").write_text("")
    ckpt = tmp_path / "checkpoints" / "local" / "single" / "vit" / "checkpoint_epoch_001.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    os.utime(ckpt, (_when("05112026_150000"),) * 2)
    msgs = []
    assert default_predictions_path(ckpt, warn=msgs.append) == logs / "test_preds_05112026_100000.npz"
    assert len(msgs) == 1 and "05112026_100000" in msgs[0]


def test_default_predictions_path_new_layout_uses_the_stamp_folder(tmp_path):
    ckpt = tmp_path / "checkpoints" / "local" / "single" / "vit" / "20112026_101500" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    msgs = []
    path = default_predictions_path(ckpt, warn=msgs.append)
    assert path == tmp_path / "logs" / "local" / "single" / "vit" / "test_preds_20112026_101500.npz"
    assert msgs == []


def test_default_predictions_path_falls_back_to_evaluation_stamp(tmp_path):
    now = datetime(2026, 11, 20, 9, 30, 5)
    msgs = []
    path = default_predictions_path(tmp_path / "ckpt.pt", now, warn=msgs.append)  # fuera
    assert path == Path("logs/test_preds_20112026_093005.npz") and msgs
    nolog = tmp_path / "checkpoints" / "local" / "single" / "vit" / "c.pt"   # sin logs
    assert default_predictions_path(nolog, now).name == "test_preds_20112026_093005.npz"


def test_threshold_source_roundtrip_and_old_files(tmp_path):
    from src.analysis.io import load_predictions
    new = _save(tmp_path / "new.npz", threshold_source="test")
    assert load_predictions(new).threshold_source == "test"
    old = _save(tmp_path / "old.npz")          # sin la clave: sigue cargando
    assert load_predictions(old).threshold_source is None


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


def test_threshold_note_names_the_source(tmp_path):
    from src.web.tabs.inspector._common import InspectorCtx, threshold_note
    for src, word in (("checkpoint", "validation"), ("test", "test set itself"),
                      (None, "not recorded")):
        path = _save(tmp_path / "logs" / f"{src}.npz", threshold_source=src)
        c = InspectorCtx.from_test(d.load_test_predictions(path, tmp_path), path)
        assert word in threshold_note(c, 0.4)


def test_default_path_for_the_results_csv_has_the_runs_stamp(tmp_path):
    # `eval --output` without a path: test_<ts>.csv, the name the web pairs with the run
    ckpt = tmp_path / "checkpoints" / "local" / "single" / "vit" / "20112026_101500" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    base = tmp_path / "logs" / "local" / "single" / "vit"
    assert default_predictions_path(ckpt, prefix="test", suffix=".csv") == base / "test_20112026_101500.csv"
    assert default_predictions_path(ckpt, prefix="val", suffix=".csv") == base / "val_20112026_101500.csv"
