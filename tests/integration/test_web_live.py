"""Pestaña Live (AppTest) sobre un logs/ sintético con un run en curso, uno terminado y otro fallido."""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "src" / "web" / "app.py")
FIXTURE = ROOT / "tests" / "fixtures" / "events_17102026_120000.jsonl"
_RUNNING, _DONE, _FAILED = "17102026_120000", "16102026_090000", "15102026_080000"
_FOLDER = "logs/local/medmnist_chest/single/vit_tiny_patch16_224"


def _line(**kw) -> str:
    return json.dumps({"v": 1, "time": 1000.0, **kw}) + "\n"


def _finished(status: str, **end) -> str:
    return (_line(type="run_start", dataset="medmnist_chest", model="vit_tiny_patch16_224",
                  mode="single", epochs=2, select_metric="f1", task="multilabel", classes=["a"])
            + _line(type="epoch_start", epoch=1, epochs=2)
            + _line(type="epoch_end", epoch=1, epochs=2, train={"loss": 0.3, "f1": 0.3},
                    val={"loss": 0.2, "f1": 0.4, "auc": 0.7}, epoch_time_s=30.0, best=0.4,
                    is_best=True)
            + _line(type="run_end", status=status, epochs_run=1, best=0.4, total_time_s=31.0,
                    **end))


@pytest.fixture(scope="module")
def live_root(tmp_path_factory):
    import streamlit as st
    from src.web.ui import helpers

    root = tmp_path_factory.mktemp("liveroot")
    folder = root / _FOLDER
    folder.mkdir(parents=True)
    for ts in (_RUNNING, _DONE, _FAILED):
        (folder / f"train_{ts}.log").write_text("x\n", encoding="utf-8")
    shutil.copy(FIXTURE, folder / f"events_{_RUNNING}.jsonl")      # mtime = now → active
    (folder / f"events_{_DONE}.jsonl").write_text(_finished("completed"), encoding="utf-8")
    (folder / f"events_{_FAILED}.jsonl").write_text(
        _finished("failed", error="CUDA out of memory"), encoding="utf-8")
    old = helpers.ROOT
    helpers.ROOT = root
    st.cache_data.clear()
    yield root
    helpers.ROOT = old
    st.cache_data.clear()


def _live(root: Path, ts: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["nav"] = "live"
    if ts:
        at.session_state["live_run"] = str(root / _FOLDER / f"events_{ts}.jsonl")
    at.run()
    return at


def _metrics(at: AppTest) -> dict:
    return {m.label: m.value for m in at.metric}


def test_live_defaults_to_the_active_run(live_root):
    at = _live(live_root)
    assert not at.exception, [e.value for e in at.exception]
    assert any(md.value == "## Live" for md in at.markdown)
    run_sel = at.selectbox(key="live_run")
    assert run_sel.value.endswith(f"events_{_RUNNING}.jsonl")        # active first
    assert any("Running" in md.value for md in at.markdown)
    m = _metrics(at)
    assert m["Epochs done"] == "2/5"
    assert m["Best AUC"] == "0.7800" and m["Val AUC"] == "0.7800"
    assert m["Val loss"] == "0.2500"
    assert at.toggle(key="live_on").value is True
    assert list(at.selectbox(key="live_interval").options) == ["2 s", "5 s", "10 s", "30 s"]


def test_live_pause_still_renders(live_root):
    at = _live(live_root)
    at.toggle(key="live_on").set_value(False).run()
    assert not at.exception
    assert _metrics(at)["Epochs done"] == "2/5"


def test_live_finished_run_shows_status(live_root):
    at = _live(live_root, _DONE)
    assert not at.exception
    assert any("Completed" in md.value for md in at.markdown)
    assert not at.error


def test_live_failed_run_shows_error(live_root):
    at = _live(live_root, _FAILED)
    assert not at.exception
    assert any("Failed" in md.value for md in at.markdown)
    assert any("CUDA out of memory" in e.value for e in at.error)


def test_live_stalled_when_file_is_old(live_root):
    folder = live_root / _FOLDER
    old = folder / "events_14102026_070000.jsonl"
    (folder / "train_14102026_070000.log").write_text("x\n", encoding="utf-8")
    shutil.copy(FIXTURE, old)
    past = time.time() - 3600
    os.utime(old, (past, past))
    try:
        at = _live(live_root, "14102026_070000")
        assert not at.exception
        assert any("Stalled" in md.value for md in at.markdown)
        assert any("interrupted" in w.value for w in at.warning)
    finally:
        old.unlink()


def test_live_without_events_shows_hint(tmp_path):
    import streamlit as st
    from src.web.ui import helpers

    (tmp_path / "logs").mkdir()
    old = helpers.ROOT
    helpers.ROOT = tmp_path
    st.cache_data.clear()
    try:
        at = _live(tmp_path)
        assert not at.exception
        assert any("No live events" in i.value for i in at.info)
    finally:
        helpers.ROOT = old
        st.cache_data.clear()
