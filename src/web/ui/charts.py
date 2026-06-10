"""Plotly chart helpers and shared styling for the dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

COLORS = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#64748b", "#ec4899", "#94a3b8"]

# ── Chart helpers ───────────────────────────────────────────────────────────────

_PLOTLY_CFG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
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
    return dict(
        title=dict(text=title, font=dict(size=13)),
        height=height,
        margin=margin if margin is not None else dict(l=50, r=16, t=48, b=40),
        # Same legend slot as _metric_fig/_overlay_fig: inside top-left,
        # translucent — clear of the outside title and the modebar.
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.65)"),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
    )


def _metric_fig(
    df: pd.DataFrame,
    col_train: str, col_val: str,
    title: str, y_label: str,
    color_train: str = COLORS[0], color_val: str = COLORS[1],
    extra_traces: list | None = None,
    height: int = 320,
) -> go.Figure:
    fig = go.Figure()
    if col_train in df.columns and df[col_train].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_train], name="Train",
            mode="lines+markers", line=dict(color=color_train, width=2), marker=dict(size=4),
        ))
    if col_val in df.columns and df[col_val].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_val], name="Val",
            mode="lines+markers", line=dict(color=color_val, width=2), marker=dict(size=4),
        ))
    for tr in (extra_traces or []):
        fig.add_trace(tr)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Epoch", yaxis_title=y_label,
        height=height, margin=dict(l=50, r=16, t=48, b=40),
        # Legend inside the plot (top-left, translucent): clear of both the
        # outside title (top-left margin) and the modebar (top-right corner).
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.65)"),
        hovermode="x unified",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
    )
    return fig


def _overlay_fig(
    dfs: list[tuple[str, pd.DataFrame]],
    col: str, title: str, y_label: str,
    height: int = 340,
) -> go.Figure:
    fig = go.Figure()
    for i, (label, df) in enumerate(dfs):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df[col],
                name=label[:30], mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2), marker=dict(size=4),
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Epoch", yaxis_title=y_label,
        height=height, margin=dict(l=50, r=16, t=48, b=40),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.65)"),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
    )
    return fig


# ── Class groups (for the confusion matrix) ──────────────────────────────────

_CLASS_GROUPS = {
    "Urban":        ([0, 1],             "#6b7280"),
    "Agricultural": ([2, 3, 4, 5, 6, 7], "#d97706"),
    "Forest":       ([8, 9, 10, 13],     "#16a34a"),
    "Scrub/grass":  ([11, 12],           "#84cc16"),
    "Bare/coastal": ([14],               "#92400e"),
    "Wetlands":     ([15, 16],           "#0891b2"),
    "Water":        ([17, 18],           "#1d4ed8"),
}
_CLASS_GROUP_COLOR: dict[int, str] = {
    idx: color for name, (idxs, color) in _CLASS_GROUPS.items() for idx in idxs
}
