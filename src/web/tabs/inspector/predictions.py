"""Predictions sub-view: gallery of the monitored samples, filters and per-sample trajectory."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.web.inspector_data import (
    SORTS, cam_overlay, filter_table, outcome_card, prediction_table, trajectory_frame,
)
from src.web.tabs.inspector._common import (
    InspectorCtx, card_grid, need_monitor, sample_cards, thumb_of,
)
from src.web.ui.charts import _show

_ALL = "All classes"
_PAGE_SIZES = [12, 24, 48, 96]


def epoch_table(c: InspectorCtx):
    """Prediction table of the selected epoch (with flips against the previous one)."""
    d = c.epoch_data()
    prev_ep = c.previous_epoch()
    prev = c.epoch_data(prev_ep) if prev_ep is not None else None
    thr = d.threshold if d.threshold is not None else 0.5
    return prediction_table(d.probs, c.monitor.targets, d.loss, c.task, c.classes, thr,
                            prev.probs if prev is not None else None), prev_ep


def render(c: InspectorCtx) -> None:
    if not need_monitor(c):
        return
    table, prev_ep = epoch_table(c)
    names = [_ALL] + c.classes
    binary = c.task == "binary"

    f = st.columns([1.3, 1.3, 1, 1.3, 1.5])
    true_sel = f[0].selectbox("True class", names, key="insp_pred_true", disabled=binary)
    pred_sel = f[1].selectbox("Predicted class", names, key="insp_pred_pred", disabled=binary)
    outcome = f[2].selectbox("Outcome", ["all", "correct", "wrong"], key="insp_pred_outcome")
    flip = f[3].selectbox(
        "Changed vs previous epoch", ["all", "changed", "to_wrong", "to_right"],
        key="insp_pred_flip", disabled=prev_ep is None,
        help="Samples whose right/wrong status differs from the previous inspected epoch."
             if prev_ep is not None else "Needs at least two inspected epochs.")
    sort = f[4].selectbox("Sort by", list(SORTS), key="insp_pred_sort")

    shown = filter_table(
        table,
        true_class=None if true_sel == _ALL else c.classes.index(true_sel),
        pred_class=None if pred_sel == _ALL else c.classes.index(pred_sel),
        outcome=outcome, flips_only=flip if prev_ep is not None else "all", sort=sort)

    total = len(shown)
    p = st.columns([1, 1, 4])
    size = p[0].selectbox("Per page", _PAGE_SIZES, index=1, key="insp_pred_size")
    pages = max(1, -(-total // size))
    page = p[1].number_input("Page", 1, pages, 1, key="insp_pred_page")
    p[2].caption(f"{total} of {len(table)} samples · epoch {c.epoch} · "
                 f"{int(table['correct'].sum())} right · green frame = right, red = wrong")
    page_rows = shown.iloc[(page - 1) * size: page * size]
    if page_rows.empty:
        st.info("No sample matches the filters.")
        return
    card_grid(sample_cards(c, page_rows))

    st.markdown("---")
    pick = st.selectbox("Inspect a sample", page_rows["pos"].tolist(), key="insp_pred_pick",
                        format_func=lambda i: f"#{i} · true: {table.at[i, 'true']} · "
                                              f"pred: {table.at[i, 'pred']}")
    _detail(c, table, int(pick))


def _detail(c: InspectorCtx, table, pos: int) -> None:
    row = table.loc[pos]
    d = c.epoch_data()
    left, right = st.columns([1, 2])
    with left:
        st.image(outcome_card(thumb_of(c, pos), bool(row["correct"]), 192), width=192)
        st.markdown(f"**True:** {row['true']}  \n**Predicted:** {row['pred']}  \n"
                    f"**Confidence:** {row['confidence']:.3f} · **Loss:** {row['loss']:.3f}")
        panel = c.monitor.panel
        if d.cam is not None and panel is not None and pos in set(np.asarray(panel).tolist()):
            k = int(np.nonzero(np.asarray(panel) == pos)[0][0])
            alpha = st.slider("Grad-CAM opacity", 0.0, 1.0, 0.5, 0.05, key="insp_pred_alpha")
            cls = int(d.cam_class[k]) if d.cam_class is not None else None
            st.image(cam_overlay(thumb_of(c, pos), d.cam[k], alpha, 192), width=192,
                     caption="Grad-CAM" + (f" of {c.classes[cls]}" if cls is not None
                                           and cls < c.n_classes else ""))
        else:
            st.caption("Not in the Grad-CAM panel.")
    with right:
        eps, probs = c.stack("probs")
        tr = trajectory_frame(eps, probs, pos, c.classes)
        truth = set(row["true_idx"])
        last = tr[tr["epoch"] == eps[-1]].nlargest(5, "prob")["class_idx"].tolist()
        fig = go.Figure()
        for ci in sorted(truth | set(last)):
            part = tr[tr["class_idx"] == ci]
            fig.add_trace(go.Scatter(
                x=part["epoch"], y=part["prob"], mode="lines+markers",
                name=("★ " if ci in truth else "") + part["class"].iloc[0],
                line={"width": 3 if ci in truth else 1.4}))
        fig.update_layout(height=320, title={"text": f"Sample #{pos}: probability by epoch "
                                                     "(★ = true label)"},
                          xaxis_title="epoch", yaxis_title="probability", yaxis_range=[0, 1])
        fig.update_xaxes(dtick=1)
        _show(fig, "insp_trajectory")
