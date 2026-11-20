"""Calibration sub-view: reliability diagram, confidence histogram and ECE / MCE / Brier by epoch."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.calibration import (
    adaptive_ece, calibration, calibration_curve_stats, per_class_calibration,
)
from src.web.inspector_stats import calibration_class_table, interpret_citl, interpret_slope
from src.web.tabs.inspector._common import InspectorCtx, _mtime, fmt, need_monitor
from src.web.tabs.inspector.curves import arrays, source_choice
from src.web.ui.charts import _show


@st.cache_data(show_spinner=False, max_entries=48)
def _calibration(dir_str: str, epoch: int, mtime: float, source: str, task: str, bins: int,
                 _probs, _targets) -> dict:
    return calibration(_probs, _targets, task, bins)


@st.cache_data(show_spinner=False, max_entries=48)
def _slope(dir_str: str, epoch: int, mtime: float, source: str, task: str, _probs,
           _targets) -> dict:
    return calibration_curve_stats(_probs, _targets, task)


@st.cache_data(show_spinner=False, max_entries=24)
def _extra(dir_str: str, epoch: int, mtime: float, source: str, task: str, bins: int, _probs,
           _targets) -> dict:
    """Adaptive (equal-mass) ECE, its class-wise version and the per-class ECE."""
    return {"adaptive": adaptive_ece(_probs, _targets, task, bins),
            "ace": adaptive_ece(_probs, _targets, task, bins, classwise=True),
            "per_class": per_class_calibration(_probs, _targets, task, bins)}


_LEVEL = {"over": st.warning, "under": st.warning, "ok": st.success, "none": st.info}


def render(c: InspectorCtx) -> None:
    if not need_monitor(c):
        return
    source = source_choice(c, "insp_calib_source")
    bins = st.slider("Bins", 5, 30, 15, key="insp_calib_bins")

    def cal(epoch: int) -> dict | None:
        a = arrays(c, epoch, source)
        if a is None:
            return None
        return _calibration(str(c.inspect_dir), epoch,
                            _mtime(c.inspect_dir / f"epoch_{epoch:03d}.npz"), source, c.task,
                            bins, *a)

    out = cal(c.epoch)
    a = arrays(c, c.epoch, source)
    mt = _mtime(c.inspect_dir / f"epoch_{c.epoch:03d}.npz")
    stats = _slope(str(c.inspect_dir), c.epoch, mt, source, c.task, *a)
    extra = _extra(str(c.inspect_dir), c.epoch, mt, source, c.task, bins, *a)
    k = st.columns(3)
    k[0].metric("ECE", fmt(out["ece"], "{:.4f}"), help="Expected calibration error "
                "(Guo et al., 2017): bin-size-weighted gap between confidence and accuracy.")
    k[1].metric("MCE", fmt(out["mce"], "{:.4f}"), help="Largest gap in any non-empty bin.")
    k[2].metric("Brier", fmt(out["brier"], "{:.4f}"))
    k = st.columns(4)
    k[0].metric("Calibration slope", fmt(stats["slope"], "{:.2f}"),
                help="Slope of the logistic recalibration logit P(y=1) = a + b·logit(p). "
                     "1 is ideal; below 1 = overconfident, above 1 = underconfident "
                     "(Van Calster et al., 2019).")
    k[1].metric("Intercept (CITL)", fmt(stats["intercept_citl"], "{:+.2f}"),
                help="Calibration-in-the-large: intercept with the slope fixed at 1. 0 is ideal; "
                     "positive = the event is more frequent than predicted.")
    k[2].metric("Adaptive ECE", fmt(extra["adaptive"]["ece"], "{:.4f}"),
                help="ECE with equal-mass bins (Nixon et al., 2019): every bin weighs the same, "
                     "so a confident network does not hide behind empty bins.")
    k[3].metric("Class-wise ECE", fmt(extra["per_class"]["sce"], "{:.4f}"),
                help="Mean over the classes of the ECE of each class probability (one-vs-rest "
                     "in multiclass).")
    level, text = interpret_slope(stats["slope"], stats["slope_se"])
    _LEVEL[level](text + " " + interpret_citl(stats["intercept_citl"]))
    st.caption("Slope and intercept: "
               + ("top-label (max probability against being right)." if c.task == "multiclass"
                  else "every (sample, class) pooled, probability against label.")
               + " Standard error of the slope: " + fmt(stats["slope_se"], "{:.2f}") + ".")
    st.caption("Multiclass: top-label confidence. Multi-label / binary: every (sample, class) is "
               "a yes/no prediction, with confidence max(p, 1-p)."
               if c.task == "multiclass" else
               "Every (sample, class) is a yes/no prediction (positive when p ≥ 0.5) with "
               "confidence max(p, 1-p), so the bins below 0.5 stay empty.")

    left, right = st.columns(2)
    with left:
        bs = [b for b in out["bins"] if b["count"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                                 line={"dash": "dot", "color": "#9AA0A6"}))
        fig.add_trace(go.Bar(x=[(b["lower"] + b["upper"]) / 2 for b in bs],
                             y=[b["accuracy"] for b in bs],
                             width=1.0 / bins * 0.9, name="accuracy", opacity=0.75,
                             customdata=[[b["count"], b["confidence"]] for b in bs],
                             hovertemplate="accuracy %{y:.3f}<br>mean confidence "
                                           "%{customdata[1]:.3f}<br>n=%{customdata[0]}"
                                           "<extra></extra>"))
        fig.update_layout(height=360, title={"text": f"Reliability diagram (epoch {c.epoch})"},
                          xaxis_title="Confidence", yaxis_title="Accuracy",
                          xaxis_range=[0, 1], yaxis_range=[0, 1.02])
        _show(fig, "insp_reliability")
    with right:
        fig = go.Figure(go.Bar(x=[(b["lower"] + b["upper"]) / 2 for b in out["bins"]],
                               y=[b["count"] for b in out["bins"]], width=1.0 / bins * 0.9))
        fig.update_layout(height=360, title={"text": "Confidence histogram"},
                          xaxis_title="Confidence", yaxis_title="Predictions", xaxis_range=[0, 1])
        _show(fig, "insp_conf_hist")

    st.markdown("##### Per class")
    st.dataframe(calibration_class_table(c.classes, extra["per_class"], stats,
                                         extra["ace"]["per_class"]).round(4),
                 hide_index=True, width="stretch")
    rows = []
    for e in c.epochs:
        if e > c.epoch:
            break
        o = cal(e)
        ae = arrays(c, e, source)
        if o is not None and ae is not None:
            sl = _slope(str(c.inspect_dir), e, _mtime(c.inspect_dir / f"epoch_{e:03d}.npz"),
                        source, c.task, *ae)
            rows.append({"epoch": e, "ECE": o["ece"], "MCE": o["mce"], "Brier": o["brier"],
                         "Slope": sl["slope"]})
    st.markdown("##### Evolution")
    if len(rows) < 2:
        st.caption("Needs at least two inspected epochs.")
        return
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for col in ("ECE", "MCE", "Brier"):   # the slope has its own scale (below)
        fig.add_trace(go.Scatter(x=df["epoch"], y=df[col], mode="lines+markers", name=col))
    fig.update_layout(height=300, xaxis_title="epoch", yaxis_title="value", hovermode="x unified")
    fig.update_xaxes(dtick=1)
    _show(fig, "insp_calib_evolution")
    sl = go.Figure(go.Scatter(x=df["epoch"], y=df["Slope"], mode="lines+markers", name="Slope"))
    sl.add_hline(y=1.0, line_dash="dot", opacity=0.6, annotation_text="ideal (1)")
    sl.update_layout(height=260, xaxis_title="epoch", yaxis_title="calibration slope",
                     title={"text": "Calibration slope by epoch (below 1 = overconfident)"})
    sl.update_xaxes(dtick=1)
    _show(sl, "insp_calib_slope")
