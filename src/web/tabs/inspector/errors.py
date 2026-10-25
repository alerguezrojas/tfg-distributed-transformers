"""Errors sub-view: worst errors, confusion matrix with the images of each cell, TP/FP/FN."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis.confusion import confusion
from src.analysis.dynamics import worst_errors
from src.web.tabs.inspector._common import InspectorCtx, card_grid, need_monitor, sample_cards
from src.web.tabs.inspector.predictions import epoch_table

_MAX_CELL_IMAGES = 24


def render(c: InspectorCtx) -> None:
    if not need_monitor(c):
        return
    table, _ = epoch_table(c)
    d = c.epoch_data()
    thr = d.threshold if d.threshold is not None else 0.5

    st.markdown("##### Worst errors")
    st.caption("Samples of the monitored subset with the highest loss at the selected epoch. "
               "Unusually high loss on a confident wrong prediction is a candidate label error.")
    k = st.slider("How many", 4, min(48, max(4, len(table))), min(12, len(table)),
                  key="insp_err_k")
    worst = worst_errors(d.probs, c.monitor.targets, d.loss, c.task, k, thr)
    card_grid(sample_cards(c, table.loc[worst["indices"]], extra="loss"))

    st.markdown("---")
    st.markdown("##### Confusion matrix")
    if c.task != "multiclass":
        thr = st.slider("Decision threshold", 0.05, 0.95, float(round(thr, 2)), 0.05,
                        key="insp_err_thr",
                        help="Prediction = probability above the threshold (epoch's optimum by default).")
    conf = confusion(d.probs, c.monitor.targets, c.task, thr)
    matrix = conf["matrix"]

    left, right = st.columns([3, 2])
    with left:
        mode = st.radio("Show", ["Counts", "Normalised by true class"], horizontal=True,
                        key="insp_err_norm")
        z = matrix.astype(float)
        if mode != "Counts":
            z = z / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        fig = go.Figure(go.Heatmap(
            z=z, x=c.classes, y=c.classes, colorscale="Blues",
            text=[[f"{v:.2f}" if mode != "Counts" else str(int(v)) for v in row] for row in z],
            texttemplate="%{text}", hovertemplate="true %{y} → predicted %{x}: %{text}<extra></extra>"))
        fig.update_layout(template="tfg", height=max(320, 28 * c.n_classes + 120),
                          xaxis_title="Predicted", yaxis_title="True", yaxis_autorange="reversed")
        event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                                key="insp_err_matrix", theme=None)
    clicked = _clicked_cell(event, c.classes)
    with right:
        st.caption("Click a cell, or choose it here:")
        i_def, j_def = clicked or (0, 0)
        ti = st.selectbox("True class", c.classes, index=i_def, key=f"insp_err_i_{i_def}")
        pj = st.selectbox("Predicted class", c.classes, index=j_def, key=f"insp_err_j_{j_def}")
    i, j = c.classes.index(ti), c.classes.index(pj)
    idx = conf["cells"][(i, j)]
    if c.task == "multilabel" or c.task == "binary":
        st.caption("Multi-label cell (i, j): samples labelled i **and** predicted j "
                   "(a sample counts in several cells).")
    st.markdown(f"**{ti} → {pj}: {len(idx)} sample(s)**"
                + (f" (showing {_MAX_CELL_IMAGES})" if len(idx) > _MAX_CELL_IMAGES else ""))
    if len(idx):
        card_grid(sample_cards(c, table.loc[idx[:_MAX_CELL_IMAGES]]))

    st.markdown("---")
    st.markdown("##### Per class: TP / FP / FN")
    rows = [{"Class": c.classes[n], "TP": p["tp"], "FP": p["fp"], "FN": p["fn"], "TN": p["tn"]}
            for n, p in enumerate(conf["per_class"])]
    st.dataframe(rows, hide_index=True, width="stretch")
    a, b = st.columns(2)
    cls = a.selectbox("Class", c.classes, key="insp_err_cls")
    kind = b.radio("Show", ["FP", "FN", "TP"], horizontal=True, key="insp_err_kind")
    sel = conf["per_class"][c.classes.index(cls)][f"{kind.lower()}_idx"]
    st.caption(f"{len(sel)} {kind} for {cls}" + (f" (showing {_MAX_CELL_IMAGES})"
                                                 if len(sel) > _MAX_CELL_IMAGES else ""))
    if len(sel):
        card_grid(sample_cards(c, table.loc[sel[:_MAX_CELL_IMAGES]]))


def _clicked_cell(event, classes: list[str]) -> tuple[int, int] | None:
    """(true index, predicted index) of the clicked heatmap cell, if any."""
    try:
        pts = event.selection.points
    except AttributeError:
        return None
    if not pts:
        return None
    p = pts[0]
    try:
        return classes.index(p["y"]), classes.index(p["x"])
    except (KeyError, ValueError):
        return None
