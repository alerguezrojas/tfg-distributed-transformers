"""Embeddings sub-view: 2-D projection of the penultimate-layer vectors."""
from __future__ import annotations

import importlib.util

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis.projection import project_2d
from src.web.inspector_data import predicted_label, primary_label
from src.web.tabs.inspector._common import InspectorCtx, _mtime, need_monitor
from src.web.tabs.inspector.predictions import epoch_table
from src.web.ui import theme
from src.web.ui.charts import _show


@st.cache_data(show_spinner=False, max_entries=32)
def _project(dir_str: str, epoch: int, mtime: float, method: str, _embeddings) -> np.ndarray:
    return project_2d(_embeddings, method)


def render(c: InspectorCtx) -> None:
    if not need_monitor(c):
        return
    d = c.epoch_data()
    if d.embeddings is None:
        st.info("This run did not save embeddings. Enable them with `monitor.embeddings: true` "
                "(default) together with `--monitor`.")
        return
    has_umap = importlib.util.find_spec("umap") is not None
    a, b = st.columns([1, 2])
    method = a.radio("Projection", ["pca", "umap"], horizontal=True, key="insp_emb_method",
                     format_func=str.upper, disabled=not has_umap,
                     help="UMAP needs the optional `umap-learn` package." if not has_umap else None)
    colour = b.radio("Colour by", ["True class", "Predicted class", "Outcome"], horizontal=True,
                     key="insp_emb_color")
    xy = _project(str(c.inspect_dir), c.epoch, _mtime(c.inspect_dir / f"epoch_{c.epoch:03d}.npz"),
                  method if has_umap else "pca", d.embeddings)
    table, _ = epoch_table(c)
    if colour == "True class":
        labels = primary_label(c.monitor.targets, c.classes)
    elif colour == "Predicted class":
        labels = predicted_label(d.probs, c.classes)
    else:
        labels = np.where(table["correct"], "right", "wrong").astype(object)

    palette = {"right": theme.GOOD, "wrong": theme.BAD, "multiple": "#9AA0A6", "none": "#C9CDD2"}
    fig = go.Figure()
    for i, name in enumerate(sorted(set(labels))):
        m = labels == name
        fig.add_trace(go.Scatter(
            x=xy[m, 0], y=xy[m, 1], mode="markers", name=str(name),
            marker={"size": 8, "opacity": 0.85,
                    "color": palette.get(name, theme.CATEGORICAL[i % len(theme.CATEGORICAL)])},
            text=[f"#{p}<br>true: {t}<br>pred: {q}" for p, t, q in
                  zip(table["pos"][m], table["true"][m], table["pred"][m])], hoverinfo="text"))
    fig.update_layout(height=520, title={"text": f"{method.upper()} of the embeddings, "
                                                 f"epoch {c.epoch}"},
                      xaxis_title="component 1", yaxis_title="component 2")
    _show(fig, "insp_embeddings")
    st.caption("Each epoch is projected on its own axes: compare how the clusters separate, not "
               "their position. Move the epoch slider at the top to follow the training. "
               + ("" if has_umap else "UMAP is not installed (`pip install umap-learn`)."))
