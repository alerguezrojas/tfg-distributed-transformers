"""Design system for the dashboard — scientific / paper-grade, deliberately plain.

One restrained accent, a muted print-friendly data palette, a neutral grotesque
(Helvetica/Arial — the typeface of scientific figures), flat surfaces (no rounded
cards, no shadows, hairline rules) and minimal chart chrome. The goal is the calm,
formal look of a journal article, not a colourful product dashboard.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Colour system ─────────────────────────────────────────────────────────────────
ACCENT = "#2272B4"          # MLflow blue — the accent
ACCENT_SOFT = "#E8F1F8"     # faint accent surface

INK = "#1A1A1A"             # near-black text
MUTED = "#6B7280"           # secondary text / axis labels
GRID = "#EEF0F2"            # gridlines (very light)
BORDER = "#E4E7EB"          # hairline borders
SURFACE = "#FFFFFF"

# MLflow-style categorical palette: one colour per run — clean, distinct, not neon.
CATEGORICAL = [
    "#2272B4",  # blue
    "#2CA02C",  # green
    "#C57B27",  # ochre
    "#9E3A47",  # maroon
    "#7E5AA2",  # purple
    "#4AA3DF",  # light blue
    "#D44E8C",  # pink
    "#6B7785",  # slate
]
SEQUENTIAL = "Blues"
DIVERGING = ["#9E3A47", "#E4E7EB", "#2272B4"]   # worse → neutral → better

GOOD, WARN, BAD = "#2E8B57", "#C57B27", "#B0413E"   # green / amber / red

_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


# ── Plotly template ─────────────────────────────────────────────────────────────────
def register_plotly_template() -> None:
    """Minimal scientific template: near-black thin axes, faint gridlines, no chartjunk."""
    axis = dict(
        gridcolor=GRID, linecolor="#9aa0a6", zeroline=False, ticks="outside",
        tickcolor="#9aa0a6", ticklen=4,
        title=dict(font=dict(size=12, color=MUTED)),
        tickfont=dict(size=11, color=MUTED),
    )
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        font=dict(family=_FONT, size=12.5, color=INK),
        title=dict(font=dict(size=13, color=INK), x=0, xanchor="left", pad=dict(b=6)),
        paper_bgcolor="white", plot_bgcolor="white",
        colorway=CATEGORICAL,
        margin=dict(l=56, r=20, t=40, b=42),
        xaxis=axis, yaxis=axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(size=11.5), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=_FONT, size=12), bgcolor="white", bordercolor=BORDER),
        colorscale=dict(sequential=SEQUENTIAL, diverging=DIVERGING),
    )
    tmpl.data.scatter = [go.Scatter(line=dict(width=1.8), marker=dict(size=4))]
    tmpl.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    pio.templates["tfg"] = tmpl
    pio.templates.default = "tfg"


register_plotly_template()


# ── CSS ───────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Inject the flat, formal layout/typography system. Call once after set_page_config."""
    st.markdown(
        """
<style>
  html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .block-container { padding-top: 2.2rem; padding-left: 2.2rem; padding-right: 2.2rem;
                     max-width: 1380px; }

  /* Typographic scale — restrained, formal, near-black (no bold-shouting). */
  [data-testid="stMarkdownContainer"] h1 { font-size: 1.5rem !important; font-weight: 600 !important;
    letter-spacing: -0.01em; color: #1A1A1A; }
  [data-testid="stMarkdownContainer"] h2 { font-size: 1.18rem !important; font-weight: 600 !important;
    margin-top: 0.7rem; color: #1A1A1A; }
  [data-testid="stMarkdownContainer"] h3 { font-size: 1.0rem !important; font-weight: 600 !important;
    margin-top: 1.0rem; color: #1A1A1A; }
  [data-testid="stMarkdownContainer"] h4 { font-size: 0.9rem !important; font-weight: 600 !important;
    color: #374151; }
  [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #2B2F35; }
  [data-testid="stCaptionContainer"] { color: #5B626B; }

  /* Metrics: plain, near-black numbers (not coloured). */
  [data-testid="stMetricValue"] { font-size: 1.2rem; font-weight: 600; color: #1A1A1A; }
  [data-testid="stMetricLabel"] { color: #5B626B; text-transform: uppercase;
    letter-spacing: 0.04em; font-size: 0.72rem; }
  [data-testid="stMetricDelta"] { font-size: 0.78rem; }

  /* Clean MLflow-style cards: white, hairline border, gentle radius + faint shadow. */
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(16,24,40,.04) !important; border: 1px solid #E4E7EB !important; }

  /* Tabs: a thin underline, restrained. */
  [data-baseweb="tab-list"] { overflow-x: auto !important; flex-wrap: nowrap !important;
    scrollbar-width: thin; gap: 0 !important; border-bottom: 1px solid #E2E4E7; }
  [data-baseweb="tab-list"]::-webkit-scrollbar { height: 3px; }
  [data-baseweb="tab-list"]::-webkit-scrollbar-thumb { background: #C9CDD2; }
  [data-baseweb="tab"] { white-space: nowrap !important; font-size: 0.84rem !important;
    padding: 0.4rem 0.85rem !important; min-width: unset !important; }

  /* Dataframes: quiet, sharp. */
  [data-testid="stDataFrame"] { border: 1px solid #E2E4E7; border-radius: 2px; }

  /* Sidebar: austere; hide the option-menu icon column for a text-only nav. */
  [data-testid="stSidebar"] { min-width: 300px; max-width: 320px;
    background: #FBFBFC; border-right: 1px solid #E2E4E7; }
  [data-testid="stSidebarUserContent"] { padding-top: 1.1rem; }
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.45rem; }
  [data-testid="stSidebar"] hr { margin: 0.5rem 0; border-color: #E2E4E7; }
  [data-testid="stSidebar"] .nav-link i, [data-testid="stSidebar"] i.bi { display: none !important; }
  [data-testid="stSidebar"] .stButton button { justify-content: flex-start; text-align: left;
    font-weight: 500; padding: 0.22rem 0.5rem; min-height: 1.9rem; }
  [data-testid="stSidebar"] .stButton button[kind="secondary"] { border: none; background: transparent; }
  [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background: #EAEEF2; color: inherit; }
  [data-testid="stSidebar"] .active-run { font-size: 0.8rem; line-height: 1.3; word-break: break-word;
    background: #E8F1F8; border: 1px solid #CFE2F0; border-radius: 6px; padding: 0.45rem 0.6rem;
    color: #1A5276; }

  /* KPI strip — flat, sharp, near-black numbers. */
  .kpi-strip { display: flex; gap: 0; margin: 0.3rem 0 0.9rem; flex-wrap: wrap;
    border: 1px solid #E4E7EB; border-radius: 8px; overflow: hidden; }
  .kpi { flex: 1; min-width: 96px; background: #fff; padding: 0.6rem 0.8rem;
    border-right: 1px solid #E2E4E7; }
  .kpi:last-child { border-right: none; }
  .kpi .v { font-size: 1.18rem; font-weight: 600; color: #1A1A1A; line-height: 1.2; }
  .kpi .l { font-size: 0.7rem; color: #5B626B; text-transform: uppercase; letter-spacing: 0.04em; }

  /* Run selector dropdown: show the FULL run label (date · [env] · model · tags)
     without clipping. The baseweb popover copies the control width, so let the
     option list grow to its content and overflow rather than truncate. */
  [data-baseweb="popover"] [role="listbox"] { min-width: max-content !important; }
  [data-baseweb="popover"] li { white-space: nowrap !important; }
</style>
""",
        unsafe_allow_html=True,
    )
