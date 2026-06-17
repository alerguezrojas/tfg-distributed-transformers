"""Plotly chart helpers for the dashboard.

Styling (font, palette, grid, margins, legend, hover) lives in the global
``tfg`` template registered in ``theme.py`` — every figure inherits it, so these
helpers only set what is chart-specific (titles, axis labels, height). No more
per-chart background / gridcolor / legend tweaks.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.ui import theme

# Ensure the design-system template is the Plotly default whenever charts are used.
theme.register_plotly_template()

# Backwards-compatible alias: the categorical palette from the design system.
COLORS = theme.CATEGORICAL

# ── Chart helpers ───────────────────────────────────────────────────────────────

# No displayModeBar key → Plotly shows the toolbar only on hover (cleaner than a
# permanently-visible bar colliding with the title/legend). PNG download stays
# available on hover, at 2× scale.
_PLOTLY_CFG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def _show(fig: go.Figure, key: str | None = None) -> None:
    """Shows a Plotly chart with a visible toolbar and PNG download."""
    cfg = dict(_PLOTLY_CFG)
    if key:
        cfg["toImageButtonOptions"] = {"format": "png", "scale": 2, "filename": key}
    st.plotly_chart(fig, use_container_width=True, config=cfg)


def _dl_csv(df: pd.DataFrame, filename: str = "data.csv", label: str = "Download CSV") -> None:
    """Download button for a DataFrame as CSV."""
    st.download_button(label, df.to_csv(index=False).encode(), file_name=filename, mime="text/csv")


def _base_layout(height: int = 320, title: str = "", margin: dict | None = None) -> dict:
    """Chart-specific layout only; the rest is inherited from the tfg template."""
    d = dict(title=dict(text=title), height=height)
    if margin is not None:
        d["margin"] = margin
    return d


def _metric_fig(
    df: pd.DataFrame,
    col_train: str, col_val: str,
    title: str, y_label: str,
    color_train: str = theme.CATEGORICAL[0], color_val: str = theme.CATEGORICAL[1],
    extra_traces: list | None = None,
    height: int = 320,
) -> go.Figure:
    fig = go.Figure()
    if col_train in df.columns and df[col_train].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_train], name="Train",
            mode="lines+markers", line=dict(color=color_train),
        ))
    if col_val in df.columns and df[col_val].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_val], name="Val",
            mode="lines+markers", line=dict(color=color_val),
        ))
    for tr in (extra_traces or []):
        fig.add_trace(tr)
    fig.update_layout(
        title=dict(text=title), xaxis_title="Epoch", yaxis_title=y_label,
        height=height, hovermode="x unified",
    )
    return fig


def _overlay_fig(
    dfs: list[tuple[str, pd.DataFrame]],
    col: str, title: str, y_label: str,
    height: int = 340,
) -> go.Figure:
    fig = go.Figure()
    n_series = 0
    for i, (label, df) in enumerate(dfs):
        if col in df.columns and df[col].notna().any():
            n_series += 1
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df[col], name=label, mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)]),
            ))
    # Many runs with long labels: legend below the plot (one row per run), and the
    # figure grows so the plot area keeps its size. Overrides the template's
    # top-right legend, which only fits a couple of short series.
    legend_px = 20 * max(n_series, 1)
    fig.update_layout(
        title=dict(text=title), xaxis_title="Epoch", yaxis_title=y_label,
        height=height + legend_px, margin=dict(b=60 + legend_px),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                    font=dict(size=11)),
    )
    return fig


# ── Class groups (for the confusion matrix) ──────────────────────────────────

# Muted, paper-grade group colours (cartographic but desaturated).
_CLASS_GROUPS = {
    "Urban":        ([0, 1],             "#5F6470"),
    "Agricultural": ([2, 3, 4, 5, 6, 7], "#9C6B3E"),
    "Forest":       ([8, 9, 10, 13],     "#4E7A6A"),
    "Scrub/grass":  ([11, 12],           "#6B7A55"),
    "Bare/coastal": ([14],               "#8A6A4A"),
    "Wetlands":     ([15, 16],           "#4E6E7A"),
    "Water":        ([17, 18],           "#3A536B"),
}
_CLASS_GROUP_COLOR: dict[int, str] = {
    idx: color for name, (idxs, color) in _CLASS_GROUPS.items() for idx in idxs
}
