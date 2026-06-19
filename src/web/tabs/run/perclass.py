"""Run results — perclass view."""
from __future__ import annotations


import plotly.graph_objects as go
import streamlit as st

from src.web.ui import theme
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (_color_f1_cell, _load_perclass)


def _per_class(ctx: DashboardContext) -> None:
    """Per-class metrics at one epoch (bars + table) and their trend across
    epochs — on a single page, no nested tabs."""
    selected_run = ctx.selected_run
    run = ctx.run
    if selected_run is None:
        st.info("Select a run in the sidebar.")
        return
    if not (run.perclass_csv_path and run.perclass_csv_path.exists()):
        st.info("No per-class data. Use `--layers confusion` to generate it.")
        return

    pcdf = _load_perclass(str(run.perclass_csv_path))
    epochs_available = sorted(pcdf["epoch"].unique().tolist())
    selected_ep = st.selectbox("Epoch", epochs_available, index=len(epochs_available) - 1,
                               format_func=lambda e: f"Epoch {e}")
    # Dot/lollipop plot (sorted by F1): one row per class, a connector showing the
    # precision↔recall↔F1 spread, and three dots. Far more legible than 57 grouped
    # bars, and the F1 dot is colour-coded by performance for an at-a-glance read.
    ep_df = pcdf[pcdf["epoch"] == selected_ep].copy().sort_values("f1", ascending=True)
    classes = ep_df["class_name"].tolist()

    conn_x: list = []
    conn_y: list = []
    for _, r in ep_df.iterrows():
        lo, hi = min(r.precision, r.recall, r.f1), max(r.precision, r.recall, r.f1)
        conn_x += [lo, hi, None]
        conn_y += [r.class_name, r.class_name, None]

    f1_colors = [theme.GOOD if v >= 0.6 else theme.WARN if v >= 0.3 else theme.BAD
                 for v in ep_df["f1"]]

    fig_pc = go.Figure()
    fig_pc.add_trace(go.Scatter(x=conn_x, y=conn_y, mode="lines",
                                line=dict(color="#CBD5E1", width=2),
                                hoverinfo="skip", showlegend=False))
    fig_pc.add_trace(go.Scatter(x=ep_df["precision"], y=classes, mode="markers",
                                name="Precision",
                                marker=dict(color=theme.CATEGORICAL[0], size=9)))
    fig_pc.add_trace(go.Scatter(x=ep_df["recall"], y=classes, mode="markers",
                                name="Recall",
                                marker=dict(color=theme.CATEGORICAL[1], size=9)))
    fig_pc.add_trace(go.Scatter(x=ep_df["f1"], y=classes, mode="markers", name="F1",
                                marker=dict(color=f1_colors, size=13, symbol="diamond",
                                            line=dict(color="white", width=1.2))))
    fig_pc.update_layout(
        title=dict(text=f"Per-class metrics — Epoch {selected_ep}"),
        xaxis_title="Score", xaxis=dict(range=[0, 1.02]),
        height=max(360, 26 * len(classes) + 90), margin=dict(l=210),
    )
    _show(fig_pc, f"per_class_ep{selected_ep}")

    with st.expander("Per-class table"):
        styled = (
            ep_df[["class_name", "f1", "precision", "recall"]]
            .style.map(_color_f1_cell, subset=["f1"])
            .format({"f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"})
        )
        st.dataframe(styled, use_container_width=True, height=280)
        _dl_csv(ep_df[["class_name", "f1", "precision", "recall"]],
                f"perclass_ep{selected_ep}.csv", "Download per-class table")

    st.markdown("#### Trend across epochs")
    classes = sorted(pcdf["class_name"].unique().tolist())
    col_sel, col_met = st.columns([3, 1])
    with col_sel:
        selected_classes = st.multiselect("Classes (max 8)", classes,
                                           default=classes[:4], max_selections=8)
    with col_met:
        metric_sel = st.radio("Metric", ["f1", "precision", "recall"])
    if selected_classes:
        fig_trend = go.Figure()
        for i, cls in enumerate(selected_classes):
            cdf = pcdf[pcdf["class_name"] == cls].sort_values("epoch")
            fig_trend.add_trace(go.Scatter(
                x=cdf["epoch"], y=cdf[metric_sel], name=cls, mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2), marker=dict(size=4),
            ))
        fig_trend.update_layout(
            **_base_layout(400, f"{metric_sel.capitalize()} per class across epochs"),
            xaxis_title="Epoch",
        )
        fig_trend.update_yaxes(range=[0, 1])
        _show(fig_trend, "class_trend")


