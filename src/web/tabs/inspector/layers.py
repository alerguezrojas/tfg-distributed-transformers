"""Layers sub-view: gradient / weight norms and update ratio per layer group, weight histograms."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.inspector_data import (
    HEALTHY_UPDATE_RATIO, LAYER_METRICS, histograms_for_epoch, layer_matrix, thin_epochs,
    update_ratio_status,
)
from src.web.tabs.inspector._common import LAYERS_HINT, InspectorCtx, fmt
from src.web.ui import theme
from src.web.ui.charts import _show

_LOG_FLOOR = -8.0      # log10 values below this are clipped (zero norms)
_RATIO_RANGE = (-6.0, 0.0)
_MAX_RIDGE = 12


def _log10(z: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(z)
    return np.where(np.isfinite(out), np.maximum(out, _LOG_FLOOR), np.nan)


def _ticks(lo: float, hi: float) -> dict:
    vals = list(range(int(np.floor(lo)), int(np.ceil(hi)) + 1))
    return {"tickvals": vals, "ticktext": [f"1e{v}" for v in vals]}


def _banded_scale(lo: float, hi: float) -> list:
    """Blue (too low) / green (healthy) / red (too high) bands for the update ratio."""
    a, b = (np.log10(v) for v in HEALTHY_UPDATE_RATIO)
    fa, fb = (a - lo) / (hi - lo), (b - lo) / (hi - lo)
    return [[0, "#4AA3DF"], [fa, "#4AA3DF"], [fa, theme.GOOD], [fb, theme.GOOD],
            [fb, theme.BAD], [1, theme.BAD]]


def render(c: InspectorCtx) -> None:
    state = c.state
    if state is None or not (state.layers or state.histograms):
        st.info("No per-layer statistics in this run. " + LAYERS_HINT)
        return
    if state.layers:
        _heatmaps(state)
    else:
        st.info("No gradient / weight statistics (no `layers` events). " + LAYERS_HINT)
    if state.histograms:
        st.markdown("---")
        _histograms(c)


def _heatmaps(state) -> None:
    n_batches = {b.get("epoch"): b.get("n_batches") for b in state.batches
                 if isinstance(b.get("n_batches"), int)}
    st.caption("Groups of layers over training time (x = fractional epoch). Colour is log10 of the value.")
    last_ratio = {}
    for metric, title in LAYER_METRICS.items():
        x, groups, z = layer_matrix(state.layers, metric, n_batches)
        if not groups:
            continue
        zl = _log10(z)
        if metric == "update_ratio":
            lo, hi = _RATIO_RANGE
            scale, bar = _banded_scale(lo, hi), {"title": "ratio", **_ticks(lo, hi)}
            last_ratio = {g: float(z[i][~np.isnan(z[i])][-1]) for i, g in enumerate(groups)
                          if (~np.isnan(z[i])).any()}
            title += (f" (healthy {HEALTHY_UPDATE_RATIO[0]:g}–{HEALTHY_UPDATE_RATIO[1]:g}: "
                      "green; blue = too low, red = too high)")
        else:
            finite = zl[np.isfinite(zl)]
            lo = float(finite.min()) if finite.size else -1.0
            hi = float(finite.max()) if finite.size else 1.0
            lo, hi = (lo - 0.5, hi + 0.5) if hi - lo < 1e-6 else (lo, hi)
            scale, bar = "Blues", {"title": "value", **_ticks(lo, hi)}
        fig = go.Figure(go.Heatmap(
            x=x, y=groups, z=zl, zmin=lo, zmax=hi, colorscale=scale, colorbar=bar,
            customdata=z, hovertemplate="%{y}<br>epoch %{x:.2f}<br>%{customdata:.3e}<extra></extra>"))
        fig.update_layout(height=max(240, 24 * len(groups) + 120), title={"text": title},
                          xaxis_title="epoch", yaxis_autorange="reversed")
        _show(fig, f"insp_layers_{metric}")
    if last_ratio:
        label = {"low": "too low", "healthy": "healthy", "high": "too high", "unknown": "—"}
        st.dataframe(pd.DataFrame({
            "Group": list(last_ratio), "Latest update ratio": [fmt(v, "{:.2e}") for v in last_ratio.values()],
            "Status": [label[update_ratio_status(v)] for v in last_ratio.values()]}),
            hide_index=True, width="stretch")


def _histograms(c: InspectorCtx) -> None:
    hist = c.state.histograms
    epochs = sorted(hist)
    st.markdown("##### Weight histograms")
    default = c.epoch if c.epoch in hist else epochs[-1]
    if len(epochs) > 1:
        epoch = st.select_slider("Epoch", epochs, value=default, key="insp_layers_epoch")
    else:
        epoch = epochs[0]
    groups = histograms_for_epoch(hist, epoch)
    if not groups:
        st.caption("No histogram for this epoch.")
        return
    group = st.selectbox("Layer group", list(groups), key="insp_layers_group")

    edges, counts = groups[group]
    centers = (edges[:-1] + edges[1:]) / 2
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=centers, y=counts, width=np.diff(edges)))
        fig.update_layout(height=320, title={"text": f"{group}, epoch {epoch}"},
                          xaxis_title="weight value", yaxis_title="count")
        _show(fig, "insp_hist_bar")
    with right:
        ridge = go.Figure()
        shown = thin_epochs(epochs, _MAX_RIDGE)
        peak = max(float(np.max(h[1])) for e in shown
                   if (h := histograms_for_epoch(hist, e).get(group)) is not None) or 1.0
        for n, e in enumerate(shown):
            h = histograms_for_epoch(hist, e).get(group)
            if h is None:
                continue
            ctr = (h[0][:-1] + h[0][1:]) / 2
            base = n * 0.6
            ridge.add_trace(go.Scatter(x=ctr, y=np.full_like(ctr, base), mode="lines",
                                       line={"width": 0}, showlegend=False, hoverinfo="skip"))
            ridge.add_trace(go.Scatter(x=ctr, y=base + h[1] / peak, mode="lines", fill="tonexty",
                                       name=f"epoch {e}", line={"width": 1}))
        ridge.update_layout(height=320, title={"text": f"{group}: weights by epoch"},
                            xaxis_title="weight value", yaxis={"showticklabels": False},
                            showlegend=False)
        _show(ridge, "insp_hist_ridge")
