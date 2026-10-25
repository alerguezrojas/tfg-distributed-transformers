"""Calibration sub-view: reliability diagram, confidence histogram and ECE / MCE / Brier by epoch."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.calibration import calibration
from src.web.tabs.inspector._common import InspectorCtx, _mtime, fmt, need_monitor
from src.web.tabs.inspector.curves import arrays, source_choice
from src.web.ui.charts import _show


@st.cache_data(show_spinner=False, max_entries=48)
def _calibration(dir_str: str, epoch: int, mtime: float, source: str, task: str, bins: int,
                 _probs, _targets) -> dict:
    return calibration(_probs, _targets, task, bins)


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
    k = st.columns(3)
    k[0].metric("ECE", fmt(out["ece"], "{:.4f}"), help="Expected calibration error "
                "(Guo et al., 2017): bin-size-weighted gap between confidence and accuracy.")
    k[1].metric("MCE", fmt(out["mce"], "{:.4f}"), help="Largest gap in any non-empty bin.")
    k[2].metric("Brier", fmt(out["brier"], "{:.4f}"))
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

    rows = []
    for e in c.epochs:
        if e > c.epoch:
            break
        o = cal(e)
        if o is not None:
            rows.append({"epoch": e, "ECE": o["ece"], "MCE": o["mce"], "Brier": o["brier"]})
    st.markdown("##### Evolution")
    if len(rows) < 2:
        st.caption("Needs at least two inspected epochs.")
        return
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for col in ("ECE", "MCE", "Brier"):
        fig.add_trace(go.Scatter(x=df["epoch"], y=df[col], mode="lines+markers", name=col))
    fig.update_layout(height=300, xaxis_title="epoch", yaxis_title="value", hovermode="x unified")
    fig.update_xaxes(dtick=1)
    _show(fig, "insp_calib_evolution")
