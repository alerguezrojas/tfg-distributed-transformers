"""Página Inspector (AppTest) sobre runs sintéticos: cada vista debe renderizar sin excepciones
para multiclass, multilabel y binary, y degradar con un mensaje cuando faltan datos."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.fixtures.make_inspector_run import make_inspector_run  # noqa: E402

APP = str(ROOT / "src" / "web" / "app.py")
VIEWS = ["System", "Performance", "Predictions", "Errors", "Curves", "Calibration",
         "Dynamics", "Embeddings", "Explain", "Layers"]


@pytest.fixture
def web_root(tmp_path):
    import streamlit as st
    from src.web.ui import helpers
    old = helpers.ROOT
    helpers.ROOT = tmp_path
    st.cache_data.clear()
    yield tmp_path
    helpers.ROOT = old
    st.cache_data.clear()


def _open(view: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["nav"] = "inspector"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.segmented_control(key="insp_view").set_value(view).run()
    return at


@pytest.mark.parametrize("task", ["multiclass", "multilabel", "binary"])
@pytest.mark.parametrize("view", VIEWS)
def test_every_view_renders(web_root, task, view):
    make_inspector_run(web_root, task=task)
    at = _open(view)
    assert not at.exception, [e.value for e in at.exception]
    # With full data no view falls back to the "no data" message.
    assert not any("No inspection data" in i.value or "No system samples" in i.value
                   or "No per-layer" in i.value or "No throughput" in i.value for i in at.info)


def test_epoch_slider_and_follow_toggle(web_root):
    make_inspector_run(web_root)
    at = _open("Predictions")
    at.toggle(key="insp_follow_25102026_120000").set_value(False).run()
    at.select_slider(key="insp_epoch_25102026_120000").set_value(1).run()
    assert not at.exception, [e.value for e in at.exception]


def test_predictions_filters_and_selection(web_root):
    make_inspector_run(web_root, task="multiclass")
    at = _open("Predictions")
    at.selectbox(key="insp_pred_outcome").set_value("wrong").run()
    at.selectbox(key="insp_pred_sort").set_value("Loss (highest first)").run()
    assert not at.exception, [e.value for e in at.exception]


def test_errors_cell_selection(web_root):
    make_inspector_run(web_root, task="multiclass")
    at = _open("Errors")
    at.selectbox(key="insp_err_cls").set_value("class_1").run()
    assert not at.exception, [e.value for e in at.exception]


def test_run_without_inspection_degrades(web_root):
    make_inspector_run(web_root, with_inspect=False)
    for view in ("Predictions", "Curves", "Explain"):
        at = _open(view)
        assert not at.exception
        assert any("--monitor" in i.value for i in at.info), view
    assert not _open("System").exception


def test_inspection_without_events(web_root):
    make_inspector_run(web_root, with_events=False)
    for view in ("System", "Performance", "Layers", "Curves"):
        at = _open(view)
        assert not at.exception, view


def test_no_runs_message(web_root):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["nav"] = "inspector"
    at.run()
    assert not at.exception
    assert any("No runs with an event stream" in i.value for i in at.info)
