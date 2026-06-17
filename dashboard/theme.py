"""Design system for the Dash dashboard — Mantine theme + a Plotly template and
chart helpers, so every figure is styled consistently (research-grade: restrained
palette, clean grid, tabular numbers).
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ── Palette ──────────────────────────────────────────────────────────────────────
ACCENT = "indigo"                 # Mantine primaryColor
INK = "#1f2933"
PALETTE = ["#4263eb", "#0ca678", "#e8590c", "#9c36b5", "#1098ad", "#f08c00", "#e64980", "#5c6470"]
GOOD, WARN, BAD = "#2f9e44", "#e8973a", "#c92a2a"
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"

MANTINE_THEME = {
    "primaryColor": ACCENT,
    "defaultRadius": "md",
    "fontFamily": FONT,
    "headings": {"fontFamily": FONT, "fontWeight": "650"},
    "colors": {
        # A calm, paper-like indigo ramp for the primary.
        "indigo": ["#edf0fb", "#dbe1f6", "#b3c0ec", "#899ce2", "#6a82da", "#5872d6",
                   "#4d69d4", "#3f59bd", "#374fa9", "#2a4196"],
    },
}

# ── Plotly template ────────────────────────────────────────────────────────────────
_GRID = "#eef1f5"


def register_template() -> None:
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        font=dict(family=FONT, size=13, color="#495057"),
        colorway=PALETTE, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=48, r=18, t=14, b=38),
        xaxis=dict(gridcolor=_GRID, linecolor="#dee2e6", zeroline=False, ticks="outside",
                   tickcolor="#dee2e6", ticklen=4, tickfont=dict(family=MONO, size=11, color="#868e96")),
        yaxis=dict(gridcolor=_GRID, linecolor="#dee2e6", zeroline=False,
                   tickfont=dict(family=MONO, size=11, color="#868e96")),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(size=12)),
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor="white", bordercolor="#dee2e6"),
    )
    pio.templates["tfg"] = tmpl
    pio.templates.default = "tfg"


register_template()


# ── Chart helpers ──────────────────────────────────────────────────────────────────
def _rgba(hexc: str, a: float) -> str:
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


def line_fig(x, series: list[dict], h: int = 280, y_range=None) -> go.Figure:
    fig = go.Figure()
    for i, s in enumerate(series):
        col = s.get("color", PALETTE[i % len(PALETTE)])
        fig.add_trace(go.Scatter(
            x=x, y=s["y"], name=s["name"], mode="lines", line=dict(width=2.4, color=col),
            fill="tozeroy" if s.get("fill") else None,
            fillcolor=_rgba(col, 0.12) if s.get("fill") else None))
    fig.update_layout(height=h, hovermode="x unified")
    if y_range:
        fig.update_yaxes(range=y_range)
    return fig


def barh_fig(cats, vals, colors=None, h: int = 300, x_max=None, label_fmt=None,
             ref=None, ref_label="", left=10) -> go.Figure:
    text = [label_fmt(v) for v in vals] if label_fmt else None
    fig = go.Figure(go.Bar(
        x=vals, y=cats, orientation="h", text=text, textposition="outside", cliponaxis=False,
        marker=dict(color=colors or PALETTE[0], line=dict(width=0)),
        marker_line_width=0))
    fig.update_traces(marker_cornerradius=4)
    fig.update_layout(height=h, showlegend=False, margin=dict(l=left, r=46, t=14, b=34))
    if x_max:
        fig.update_xaxes(range=[0, x_max])
    if ref is not None:
        fig.add_vline(x=ref, line_dash="dash", line_color="#868e96",
                      annotation_text=ref_label, annotation_position="top")
    return fig


def donut_fig(labels, values, h: int = 300) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62,
                           marker=dict(colors=PALETTE, line=dict(color="white", width=2))))
    fig.update_traces(textinfo="none", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>")
    fig.update_layout(height=h, legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
    return fig


def dumbbell_fig(classes, A, B, a_label, b_label, h: int = 420) -> go.Figure:
    fig = go.Figure()
    for sign, col in ((1, GOOD), (-1, BAD)):
        xs, ys = [], []
        for c, a, b in zip(classes, A, B):
            if (b - a > 0.01) == (sign > 0) and abs(b - a) > 0.01:
                xs += [a, b, None]; ys += [c, c, None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=2.4),
                                     hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=A, y=classes, mode="markers", name=a_label,
                             marker=dict(color="#adb5bd", size=10)))
    fig.add_trace(go.Scatter(x=B, y=classes, mode="markers", name=b_label,
                             marker=dict(color=PALETTE[0], size=11)))
    fig.update_layout(height=h, margin=dict(l=200, r=20, t=14, b=44))
    fig.update_xaxes(range=[0, 1.02], title="F1")
    return fig


def scaling_fig(ns, speedups, h: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=ns, name="Ideal (linear)", mode="lines",
                             line=dict(dash="dash", color="#adb5bd", width=1.6)))
    fig.add_trace(go.Scatter(x=ns, y=speedups, name="Predicted", mode="lines+markers",
                             line=dict(width=2.6, color=PALETTE[0]), marker=dict(size=7),
                             fill="tozeroy", fillcolor=_rgba(PALETTE[0], 0.08)))
    fig.update_layout(height=h, hovermode="x unified")
    fig.update_xaxes(title="GPUs"); fig.update_yaxes(title="speedup ×")
    return fig


def treemap_fig(labels, values, h: int = 340) -> go.Figure:
    fig = go.Figure(go.Treemap(
        labels=labels, parents=[""] * len(labels), values=values,
        marker=dict(colors=values, colorscale="Tealgrn", line=dict(color="white", width=1.5)),
        texttemplate="%{label}<br>%{value:,}", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} patches<extra></extra>", tiling=dict(pad=2)))
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=4, b=0))
    return fig
