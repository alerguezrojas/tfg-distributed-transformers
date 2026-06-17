"""Design system for the dashboard — one source of truth for colour, type and
chart styling, so every view looks coherent instead of ad-hoc.

Three pieces:
  • a colour system (one accent + neutrals + categorical / sequential / diverging
    palettes) — no more loose hex lists per chart;
  • a single Plotly template ("tfg") registered as the default, so every figure
    inherits the same font, grid, margins and palette automatically;
  • inject_css() — Inter typography, spacing rhythm, tables and sidebar polish.

Direction: clean light / editorial (reproduces well on a projector and in the
report PDF), deep-blue accent from the project.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Colour system ─────────────────────────────────────────────────────────────────
ACCENT = "#1A5276"          # project deep blue — primary accent
ACCENT_SOFT = "#EAF1F6"     # tinted background for accent surfaces

INK = "#0F172A"             # primary text
MUTED = "#64748B"           # secondary text / axis titles
GRID = "#EEF2F6"            # gridlines
BORDER = "#E2E8F0"          # axis lines, table borders
SURFACE = "#F8FAFC"         # subtle panel background

# Muted, harmonious categorical palette anchored on the accent (editorial, not neon).
CATEGORICAL = [
    "#1A5276",  # deep blue
    "#C77D11",  # ochre
    "#2C7873",  # teal
    "#9B2D30",  # brick red
    "#5D5179",  # muted violet
    "#4E7A51",  # sage
    "#8295A8",  # slate
    "#B05A3C",  # terracotta
]
SEQUENTIAL = "Blues"                              # intensity / heatmaps
DIVERGING = ["#9B2D30", "#EEF2F6", "#2E7D32"]     # worse → neutral → better

# Semantic (verdicts, deltas)
GOOD, WARN, BAD = "#2E7D32", "#C77D11", "#9B2D30"

_FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"


# ── Plotly template ─────────────────────────────────────────────────────────────────

def register_plotly_template() -> None:
    """Register the 'tfg' template and make it the default for every figure."""
    axis = dict(
        gridcolor=GRID, linecolor=BORDER, zerolinecolor=BORDER, zeroline=False,
        ticks="outside", tickcolor=BORDER, ticklen=4,
        title=dict(font=dict(size=12, color=MUTED)),
        tickfont=dict(size=11, color=MUTED),
    )
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        font=dict(family=_FONT, size=13, color=INK),
        title=dict(font=dict(size=14, color=INK), x=0, xanchor="left", pad=dict(b=6)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=CATEGORICAL,
        margin=dict(l=56, r=20, t=48, b=44),
        xaxis=axis, yaxis=axis,
        # Legend on top-right so it never collides with the left-aligned title
        # nor the modebar — replaces the old translucent-inside-plot hack.
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=12),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=_FONT, size=12), bgcolor="white",
                        bordercolor=BORDER),
        colorscale=dict(sequential=SEQUENTIAL, diverging=DIVERGING),
    )
    # Cleaner default marker/line styling for the common trace types.
    tmpl.data.scatter = [go.Scatter(line=dict(width=2.4), marker=dict(size=5))]
    tmpl.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    pio.templates["tfg"] = tmpl
    pio.templates.default = "tfg"


# ── CSS ───────────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    """Inject the typography + layout system. Call once after set_page_config."""
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .block-container { padding-top: 2.2rem; padding-left: 2rem; padding-right: 2rem;
                     max-width: 1400px; }

  /* Typographic scale — tight, editorial. Scoped to beat Streamlit's own rules. */
  [data-testid="stMarkdownContainer"] h1 { font-size: 1.5rem !important; font-weight: 700 !important;
    letter-spacing: -0.015em; color: #0F172A; }
  [data-testid="stMarkdownContainer"] h2 { font-size: 1.18rem !important; font-weight: 650 !important;
    letter-spacing: -0.01em; margin-top: 0.6rem; color: #0F172A; }
  [data-testid="stMarkdownContainer"] h3 { font-size: 1.0rem !important; font-weight: 600 !important;
    margin-top: 0.9rem; color: #1E293B; }
  [data-testid="stMarkdownContainer"] h4 { font-size: 0.88rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.04em; color: #64748B; }
  [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #334155; }
  [data-testid="stCaptionContainer"] { color: #64748B; }
  code, kbd, pre { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.82rem; }

  /* Metrics: lighter, less "template card". */
  [data-testid="stMetricValue"] { font-size: 1.2rem; font-weight: 650; color: #0F172A; }
  [data-testid="stMetricLabel"] { color: #64748B; text-transform: uppercase;
    letter-spacing: 0.03em; font-size: 0.72rem; }
  [data-testid="stMetricDelta"] { font-size: 0.78rem; }

  /* Bordered containers (st.container(border=True)) → clean cards. */
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 0.6rem; }

  /* One row of tabs, scrollable, thin underline. */
  [data-baseweb="tab-list"] { overflow-x: auto !important; flex-wrap: nowrap !important;
    scrollbar-width: thin; gap: 0 !important; border-bottom: 1px solid #E2E8F0; }
  [data-baseweb="tab-list"]::-webkit-scrollbar { height: 3px; }
  [data-baseweb="tab-list"]::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
  [data-baseweb="tab"] { white-space: nowrap !important; font-size: 0.84rem !important;
    padding: 0.4rem 0.8rem !important; min-width: unset !important; }

  /* Dataframes: quieter chrome. */
  [data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 0.5rem; }

  /* Sidebar: compact, borderless menu. */
  [data-testid="stSidebar"] { min-width: 300px; max-width: 320px; }
  [data-testid="stSidebarUserContent"] { padding-top: 1.1rem; }
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.45rem; }
  [data-testid="stSidebar"] hr { margin: 0.5rem 0; border-color: #E2E8F0; }
  [data-testid="stSidebar"] .stButton button { justify-content: flex-start; text-align: left;
    font-weight: 500; padding: 0.22rem 0.5rem; min-height: 1.9rem; }
  [data-testid="stSidebar"] .stButton button div,
  [data-testid="stSidebar"] .stButton button p { justify-content: flex-start; text-align: left; width: 100%; }
  [data-testid="stSidebar"] .stButton button[kind="secondary"] { border: none; background: transparent; }
  [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background: rgba(26,82,118,0.08); color: inherit; }
  [data-testid="stSidebar"] .active-run { font-size: 0.8rem; line-height: 1.3; word-break: break-word;
    background: #EAF1F6; border: 1px solid #D7E3EC; border-radius: 0.45rem; padding: 0.45rem 0.6rem;
    color: #1A5276; font-weight: 500; }

  /* Compact KPI strip — denser and calmer than st.metric cards. */
  .kpi-strip { display: flex; gap: 0.6rem; margin: 0.3rem 0 0.9rem; flex-wrap: wrap; }
  .kpi { flex: 1; min-width: 96px; background: #fff; border: 1px solid #E2E8F0;
         border-radius: 0.55rem; padding: 0.55rem 0.7rem; }
  .kpi .v { font-size: 1.2rem; font-weight: 700; color: #1A5276; line-height: 1.2;
            letter-spacing: -0.01em; }
  .kpi .l { font-size: 0.7rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.03em; }
</style>
""",
        unsafe_allow_html=True,
    )
