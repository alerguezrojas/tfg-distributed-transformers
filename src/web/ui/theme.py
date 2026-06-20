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


# ── Dark-mode palette (the chart/plot surfaces; CSS handles the Streamlit chrome) ──
_DARK = dict(ink="#E6E8EB", muted="#9AA0A6", grid="#2A2F3A", line="#5B626B",
             paper="#161A23", border="#2A2F3A")


# ── Plotly template ─────────────────────────────────────────────────────────────────
def register_plotly_template(mode: str = "light") -> None:
    """Minimal scientific template — light or dark, selected by ``mode``."""
    dark = mode == "dark"
    ink = _DARK["ink"] if dark else INK
    muted = _DARK["muted"] if dark else MUTED
    grid = _DARK["grid"] if dark else GRID
    line = _DARK["line"] if dark else "#9aa0a6"
    paper = _DARK["paper"] if dark else "white"
    border = _DARK["border"] if dark else BORDER
    axis = dict(
        gridcolor=grid, linecolor=line, zeroline=False, ticks="outside",
        tickcolor=line, ticklen=4,
        title=dict(font=dict(size=12, color=muted)),
        tickfont=dict(size=11, color=muted),
    )
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        font=dict(family=_FONT, size=12.5, color=ink),
        title=dict(font=dict(size=13, color=ink), x=0, xanchor="left", pad=dict(b=6)),
        paper_bgcolor=paper, plot_bgcolor=paper,
        colorway=CATEGORICAL,
        margin=dict(l=56, r=20, t=40, b=42),
        xaxis=axis, yaxis=axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(size=11.5), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=_FONT, size=12), bgcolor=paper, bordercolor=border),
        colorscale=dict(sequential=SEQUENTIAL, diverging=DIVERGING),
    )
    tmpl.data.scatter = [go.Scatter(line=dict(width=1.8), marker=dict(size=4))]
    tmpl.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    pio.templates["tfg"] = tmpl
    pio.templates.default = "tfg"


register_plotly_template()


# ── CSS ───────────────────────────────────────────────────────────────────────────
def inject_css(mode: str = "light") -> None:
    """Inject the flat, formal layout/typography system. Call once after
    set_page_config. ``mode='dark'`` appends an override layer for dark mode."""
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

  /* Sidebar: austere; hide the option-menu icon column for a text-only nav.
     Wide enough that the longest run label fits on one line (like the Overview
     table), e.g. "10/06/2026 21:43 [kaggle] vit_large [model_parallel]". */
  [data-testid="stSidebar"] { min-width: 470px; max-width: 470px;
    background: #FBFBFC; border-right: 1px solid #E2E4E7; }
  [data-testid="stSidebarUserContent"] { padding-top: 1.1rem; }
  /* Breathing room between sidebar widgets so labels never touch the box above. */
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.7rem; }
  [data-testid="stSidebar"] hr { margin: 0.5rem 0; border-color: #E2E4E7; }
  [data-testid="stSidebar"] .nav-link i, [data-testid="stSidebar"] i.bi { display: none !important; }
  /* Sidebar buttons = the run-list rows (nav is the option_menu iframe, not a
     button). Left-aligned, compact, and each row stays on ONE line — the inner
     label element is a div, so target the button and all its children. */
  /* Run-list rows: the element containers are ~content width and get centred in
     the list; force them to fill the width (margin 0, max-width none) and push
     the label flush against the left edge. */
  [data-testid="stSidebar"] [data-testid="stElementContainer"],
  [data-testid="stSidebar"] .stButton {
    width: 100% !important; max-width: 100% !important; margin: 0 !important; }
  [data-testid="stSidebar"] .stButton button {
    width: 100% !important; justify-content: flex-start !important; text-align: left !important;
    font-weight: 500; font-size: 0.8rem; padding: 0.2rem 0.45rem; min-height: 1.7rem; }
  /* The button's inner content div is display:flex; justify-content:center —
     that is what centres the label. Force it to start so the text sits left. */
  [data-testid="stSidebar"] .stButton button > * {
    width: 100% !important; justify-content: flex-start !important; text-align: left !important; }
  [data-testid="stSidebar"] .stButton button,
  [data-testid="stSidebar"] .stButton button * {
    white-space: nowrap !important; overflow: hidden; text-overflow: ellipsis; }
  [data-testid="stSidebar"] .stButton button[kind="secondary"] { border: none; background: transparent; }
  [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background: #EAEEF2; color: inherit; }
  [data-testid="stSidebar"] .active-run { font-size: 0.8rem; line-height: 1.3; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
    background: #E8F1F8; border: 1px solid #CFE2F0; border-radius: 6px; padding: 0.45rem 0.6rem;
    color: #1A5276; margin-bottom: 0.4rem; }

  /* KPI strip — flat, sharp, near-black numbers. */
  .kpi-strip { display: flex; gap: 0; margin: 0.3rem 0 0.9rem; flex-wrap: wrap;
    border: 1px solid #E4E7EB; border-radius: 8px; overflow: hidden; }
  .kpi { flex: 1; min-width: 96px; background: #fff; padding: 0.6rem 0.8rem;
    border-right: 1px solid #E2E4E7; }
  .kpi:last-child { border-right: none; }
  .kpi .v { font-size: 1.18rem; font-weight: 600; color: #1A1A1A; line-height: 1.2; }
  .kpi .l { font-size: 0.7rem; color: #5B626B; text-transform: uppercase; letter-spacing: 0.04em; }
</style>
""",
        unsafe_allow_html=True,
    )
    if mode == "dark":
        _inject_dark_css()


def _inject_dark_css() -> None:
    """Dark-mode override layer over the light base (Streamlit's chrome + our
    components). Charts switch via the Plotly template; the st.dataframe grid
    stays light (its theming is config-time, not CSS-controllable)."""
    st.markdown(
        """
<style>
  /* App surfaces + base text */
  [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .stApp {
    background-color: #0E1117 !important; color: #E6E8EB !important; }
  [data-testid="stHeader"] { background: transparent !important; }

  /* Typography */
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3 { color: #E6E8EB !important; }
  [data-testid="stMarkdownContainer"] h4 { color: #C9CDD2 !important; }
  [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #C9CDD2 !important; }
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #9AA0A6 !important; }

  /* Metrics */
  [data-testid="stMetricValue"] { color: #E6E8EB !important; }
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #9AA0A6 !important; }

  /* Cards / bordered containers */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: #161A23 !important; border-color: #2A2F3A !important;
    box-shadow: 0 1px 2px rgba(0,0,0,.35) !important; }

  /* Tabs */
  [data-baseweb="tab-list"] { border-bottom-color: #2A2F3A !important; }
  [data-baseweb="tab"] { color: #C9CDD2 !important; }

  /* Inputs / selects */
  [data-baseweb="select"] > div, [data-baseweb="input"] input,
  [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
  textarea { background: #161A23 !important; color: #E6E8EB !important;
    border-color: #2A2F3A !important; }
  [data-testid="stDataFrame"] { border-color: #2A2F3A !important; }
  [data-testid="stExpander"] { border-color: #2A2F3A !important; }
  [data-testid="stExpander"] details { background: #161A23 !important; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #14181F !important; border-right-color: #2A2F3A !important; }
  [data-testid="stSidebar"] hr { border-color: #2A2F3A !important; }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #C9CDD2 !important; }
  [data-testid="stSidebar"] .stButton button { color: #C9CDD2 !important; }
  [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover { background: #20262F !important; }
  [data-testid="stSidebar"] .active-run {
    background: #18293B !important; border-color: #2A4866 !important; color: #BBD6F0 !important; }

  /* KPI strip */
  .kpi-strip { border-color: #2A2F3A !important; }
  .kpi { background: #161A23 !important; border-right-color: #2A2F3A !important; }
  .kpi .v { color: #E6E8EB !important; }
  .kpi .l { color: #9AA0A6 !important; }
</style>
""",
        unsafe_allow_html=True,
    )
