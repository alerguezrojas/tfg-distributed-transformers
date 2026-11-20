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
VIEWS = ["System", "Performance", "Predictions", "Errors", "Curves", "Clinical", "Calibration",
         "Dynamics", "Embeddings", "Explain", "Layers", "Alerts"]
ALERTS = [
    {"level": "warning", "code": "overfitting", "message": "val loss rises", "epoch": 2,
     "batch": None, "value": 0.9, "threshold": 0.5},
    {"level": "critical", "code": "nan_loss", "message": "loss is NaN", "epoch": 3, "batch": 12,
     "value": None, "threshold": None},
]


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
                   or "No per-layer" in i.value or "No throughput" in i.value
                   or "has no operating point" in i.value for i in at.info)


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
    for view in ("Predictions", "Curves", "Clinical", "Explain"):
        at = _open(view)
        assert not at.exception
        assert any("--monitor" in i.value for i in at.info), view
    assert not _open("System").exception


def test_inspection_without_events(web_root):
    make_inspector_run(web_root, with_events=False)
    for view in ("System", "Performance", "Layers", "Curves", "Clinical", "Calibration", "Alerts"):
        at = _open(view)
        assert not at.exception, view
    assert any("does not have" in i.value for i in _open("Alerts").info)


def test_no_runs_message(web_root):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["nav"] = "inspector"
    at.run()
    assert not at.exception
    assert any("No runs with an event stream" in i.value for i in at.info)


def _texts(at):
    return [e.value for e in (*at.error, *at.warning, *at.success, *at.info, *at.caption,
                              *at.markdown)]


@pytest.mark.parametrize("task", ["multiclass", "multilabel", "binary"])
def test_curves_confidence_intervals_and_controls(web_root, task):
    make_inspector_run(web_root, task=task)
    at = _open("Curves")
    assert any("percentile bootstrap, n=1,000" in t for t in _texts(at))
    assert any(m.label == "Macro AUC" and m.delta for m in at.metric)
    at.selectbox(key="insp_ci_n").set_value(200).run()
    assert any("n=200" in t for t in _texts(at)) and not at.exception
    at.toggle(key="insp_ci_cls").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("intervals: 95% CI" in m.value for m in at.markdown)
    at.toggle(key="insp_ci_on").set_value(False).run()
    assert not at.exception and any("Intervals off" in t for t in _texts(at))


@pytest.mark.parametrize("task", ["multiclass", "multilabel", "binary"])
def test_clinical_criteria_and_prevalence(web_root, task):
    make_inspector_run(web_root, task=task)
    at = _open("Clinical")
    assert {m.label for m in at.metric} >= {"Sensitivity", "Specificity", "PPV", "NPV", "F1"}
    for crit in ("sensitivity", "youden", "validation", "fixed"):
        if crit == "validation" and task == "multiclass":
            continue
        at.radio(key="insp_clin_crit").set_value(crit).run()
        assert not at.exception, (crit, [e.value for e in at.exception])
    at.toggle(key="insp_clin_prev_on").set_value(True).run()
    at.number_input(key="insp_clin_prev").set_value(1.0).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("set by you" in t for t in _texts(at))
    at.selectbox(key="insp_clin_n").set_value(0).run()
    assert not at.exception and not any("percentile bootstrap" in t for t in _texts(at))


def test_clinical_sensitivity_target_is_reached(web_root):
    make_inspector_run(web_root, task="multilabel")
    at = _open("Clinical")
    at.radio(key="insp_clin_crit").set_value("sensitivity").run()
    at.select_slider(key="insp_clin_target").set_value(0.95).run()
    assert not at.exception
    sens = next(m for m in at.metric if m.label == "Sensitivity")
    assert float(sens.value) >= 0.95
    assert any("conditional on it" in t for t in _texts(at))


def test_clinical_class_selection(web_root):
    make_inspector_run(web_root, task="multiclass")
    at = _open("Clinical")
    at.selectbox(key="insp_clin_cls").set_value("class_2").run()
    assert not at.exception
    assert any("class_2" in m.value for m in at.markdown)


@pytest.mark.parametrize("task", ["multiclass", "multilabel", "binary"])
def test_calibration_slope_and_class_table(web_root, task):
    make_inspector_run(web_root, task=task)
    at = _open("Calibration")
    labels = {m.label for m in at.metric}
    assert {"Calibration slope", "Intercept (CITL)", "Adaptive ECE", "Class-wise ECE"} <= labels
    assert any(("Slope" in t and "overconfident" in t) or "compatible with 1" in t
               or "underconfident" in t or "undefined" in t for t in _texts(at))
    assert any(d.columns.tolist()[:2] == ["Class", "Positives"] for d in
               (df.value for df in at.dataframe))
    at.slider(key="insp_calib_bins").set_value(8).run()
    assert not at.exception, [e.value for e in at.exception]


def test_subgroups_only_with_groups_file(web_root):
    make_inspector_run(web_root)
    at = _open("Predictions")
    assert "Subgroups" not in at.segmented_control(key="insp_view").options
    assert not at.exception


@pytest.mark.parametrize("task", ["multiclass", "multilabel", "binary"])
def test_subgroups_view(web_root, task):
    make_inspector_run(web_root, task=task, with_groups=True, m=120)
    at = _open("Predictions")
    assert "Subgroups" in at.segmented_control(key="insp_view").options
    at.segmented_control(key="insp_view").set_value("Subgroups").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(str(df.value["Group"].iloc[0]) == "All samples" for df in at.dataframe)
    for var in ("age", "centre"):
        at.selectbox(key="insp_grp_var").set_value(var).run()
        assert not at.exception, (var, [e.value for e in at.exception])
    at.selectbox(key="insp_grp_n").set_value(0).run()
    assert not at.exception and any("No intervals" in t for t in _texts(at))
    at.selectbox(key="insp_grp_metric").set_value("sensitivity").run()
    assert not at.exception


def test_alerts_banner_and_view(web_root):
    make_inspector_run(web_root, alerts=ALERTS)
    at = _open("Alerts")
    assert not at.exception, [e.value for e in at.exception]
    assert any("CRITICAL" in e.value and "nan_loss" in e.value and "epoch 3" in e.value
               for e in at.error)
    assert any("overfitting" in w.value and "epoch 2" in w.value for w in at.warning)
    assert [m.value for m in at.metric if m.label == "Critical"] == ["1"]
    at.multiselect(key="insp_alert_level").set_value(["critical"]).run()
    assert not at.exception
    # the banner is in the header: it is there in every view
    assert any("nan_loss" in e.value for e in _open("Curves").error)


def test_alerts_view_without_alerts(web_root):
    make_inspector_run(web_root)
    at = _open("Alerts")
    assert not at.exception and any("No alerts" in s.value for s in at.success)
    assert not at.error


def test_layers_histogram_kind(web_root):
    make_inspector_run(web_root, grad_hist=True)
    at = _open("Layers")
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="insp_hist_kind").options == ["Weights", "Gradients"]
    at.radio(key="insp_hist_kind").set_value("Gradients").run()
    assert not at.exception
    assert any("Gradient histograms" in m.value for m in at.markdown)


def test_layers_without_gradient_histograms_has_no_selector(web_root):
    make_inspector_run(web_root)
    at = _open("Layers")
    assert not at.exception
    assert not any(r.key == "insp_hist_kind" for r in at.radio)


def test_live_shows_the_alert_banner(web_root):
    make_inspector_run(web_root, alerts=ALERTS)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["nav"] = "live"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("CRITICAL" in e.value and "nan_loss" in e.value for e in at.error)
    assert any("overfitting" in w.value for w in at.warning)


def test_live_without_alerts_has_no_banner(web_root):
    make_inspector_run(web_root)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["nav"] = "live"
    at.run()
    assert not at.exception and not at.error
