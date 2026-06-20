"""Run results — perclass view."""
from __future__ import annotations


import plotly.graph_objects as go
import streamlit as st

from src.web.ui import theme
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (_color_f1_cell, _load_perclass, _load_val_support,
                                _dataset_meta_path)


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
    # Annotated heatmap (classes × precision/recall/F1): colour = score (red→amber→
    # green, same thresholds as the F1 verdict), value printed in each cell. Reads
    # at a glance which classes are weak AND whether it's precision or recall — the
    # whole point of the per-class view — far clearer than 3 overlaid dot series.
    ep_df = pcdf[pcdf["epoch"] == selected_ep].copy().sort_values("f1", ascending=True)
    # Support (val-set frequency of each class) — a dataset property, so it shows
    # for runs already trained without retraining. Used as hover context + a table
    # column: it tells whether a low F1 is on a rare class or a real failure.
    meta = _dataset_meta_path()
    support = _load_val_support(meta) if meta else None
    if support:
        ep_df["support"] = ep_df["class_name"].map(support).fillna(0).astype(int)

    classes = ep_df["class_name"].tolist()
    z = [[p, r, f] for p, r, f in zip(ep_df["precision"], ep_df["recall"], ep_df["f1"])]
    # Red (low) → amber (~0.3) → green (≥0.6), matching theme.GOOD/WARN/BAD.
    scale = [[0.0, theme.BAD], [0.3, theme.WARN], [0.6, theme.GOOD], [1.0, theme.GOOD]]

    if support:
        cdata = [[int(s)] * 3 for s in ep_df["support"]]
        hover = ("<b>%{y}</b><br>%{x}: %{z:.3f}"
                 "<br>val support: %{customdata} patches<extra></extra>")
    else:
        cdata, hover = None, "<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>"

    fig_pc = go.Figure(go.Heatmap(
        z=z, x=["Precision", "Recall", "F1"], y=classes, customdata=cdata,
        colorscale=scale, zmin=0.0, zmax=1.0, xgap=2, ygap=2,
        texttemplate="%{z:.2f}", textfont=dict(size=10, color="white"),
        colorbar=dict(title="Score", thickness=12, len=0.8),
        hovertemplate=hover,
    ))
    fig_pc.update_layout(
        title=dict(text=f"Per-class metrics — Epoch {selected_ep}"),
        height=max(360, 24 * len(classes) + 110),
        margin=dict(l=210, t=48, r=10, b=10),
        xaxis=dict(side="top"),
    )
    _show(fig_pc, f"per_class_hm_ep{selected_ep}")
    st.caption("Colour = score (green good · amber middling · red poor). Compare the "
               "Precision and Recall cells of a weak class to see *why* its F1 is low: "
               "low recall → it misses that class; low precision → it over-predicts it. "
               "Hover a cell (or see the table) for **support** — the class's frequency "
               "in the validation set: a low F1 on a tiny-support class is a rare class, "
               "not a broken model.")

    with st.expander("Per-class table"):
        cols = ["class_name", "f1", "precision", "recall"]
        if "support" in ep_df.columns:
            cols.append("support")          # val-set frequency of each class
        styled = (
            ep_df[cols].sort_values("f1", ascending=False)
            .style.map(_color_f1_cell, subset=["f1"])
            .format({"f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"})
        )
        st.dataframe(styled, use_container_width=True, height=280)
        _dl_csv(ep_df[cols], f"perclass_ep{selected_ep}.csv", "Download per-class table")

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


