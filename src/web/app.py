"""Streamlit web dashboard — Training Dashboard (English).

Thin orchestrator: sets the page config, builds the sidebar, assembles the
shared context, and dispatches to the per-tab render modules under
``src/web/tabs/``. Reusable chart helpers and cached loaders live in
``src/web/ui/``. This keeps every concern (sidebar, each tab, charts, loaders)
in its own module — single responsibility instead of one 3000-line file.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from streamlit_option_menu import option_menu

from src.web.ui import i18n
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import _get_runs
from src.web.tabs import (
    home,
    run as run_tab,
    comparison,
    feasibility,
    data_models,
)

# ── Page configuration ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Training Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional Spanish view (English is the default). Installed before any rendering
# so the whole UI — sidebar included — is translated on this run.
_lang = st.session_state.get("_lang", "en")
i18n.install(_lang)

st.markdown("""
<style>
  [data-testid="stSidebar"] { min-width: 300px; max-width: 320px; }
  .block-container { padding-top: 2.4rem; padding-left: 1.5rem; padding-right: 1.5rem; }
  /* Headings: Streamlit's own rules win over bare element selectors, so scope
     to the markdown container and force a compact, professional scale. */
  [data-testid="stMarkdownContainer"] h1 { font-size: 1.45rem !important; font-weight: 650 !important; }
  [data-testid="stMarkdownContainer"] h2 {
    font-size: 1.25rem !important; font-weight: 650 !important;
    margin-top: 0.4rem; padding-bottom: 0.2rem;
  }
  [data-testid="stMarkdownContainer"] h3 {
    font-size: 1.02rem !important; font-weight: 600 !important; margin-top: 0.8rem;
  }
  [data-testid="stMarkdownContainer"] h4 { font-size: 0.92rem !important; font-weight: 600 !important; }
  [data-testid="stMetricValue"] { font-size: 1.1rem; }
  [data-testid="stMetricLabel"] { opacity: 0.75; }
  [data-baseweb="tab-list"] {
    overflow-x: auto !important; flex-wrap: nowrap !important;
    scrollbar-width: thin; gap: 0 !important;
    border-bottom: 1px solid #e5e7eb;
  }
  [data-baseweb="tab-list"]::-webkit-scrollbar { height: 3px; }
  [data-baseweb="tab-list"]::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
  [data-baseweb="tab"] {
    white-space: nowrap !important; font-size: 0.82rem !important;
    padding-left: 0.75rem !important; padding-right: 0.75rem !important;
    min-width: unset !important;
  }
  /* Sidebar: compact — everything (nav + run selector + language) must fit
     without scrolling. Tighter gaps, slim dividers, less top padding. */
  [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.45rem; }
  [data-testid="stSidebar"] hr { margin: 0.45rem 0; }
  /* Sidebar navigation: borderless menu items; the active page keeps the
     filled accent (primary button). */
  [data-testid="stSidebar"] .stButton button {
    justify-content: flex-start; text-align: left; font-weight: 500;
    padding-top: 0.22rem; padding-bottom: 0.22rem; min-height: 1.9rem;
  }
  /* The label lives in a nested <p>; align it too or the text stays centered. */
  [data-testid="stSidebar"] .stButton button div { justify-content: flex-start; }
  [data-testid="stSidebar"] .stButton button p { text-align: left; width: 100%; }
  [data-testid="stSidebar"] .stButton button[kind="secondary"] {
    border: none; background: transparent;
  }
  [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background: rgba(26, 82, 118, 0.08); color: inherit;
  }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    margin-top: 0.4rem; letter-spacing: 0.04em; opacity: 0.7;
  }
  /* Active-run box in the sidebar: full name, wraps, small. */
  [data-testid="stSidebar"] .active-run {
    font-size: 0.8rem; line-height: 1.3; word-break: break-word;
    background: #f1f5f9; border-radius: 0.4rem; padding: 0.4rem 0.55rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

runs = _get_runs()

# Single-level sidebar navigation (icon menu). Each entry renders one page
# module full-width. Pages keep at most ONE row of tabs inside — never 3 levels.
# System was removed: the live hardware monitor was not useful with the Kaggle
# workflow, and "Import runs" now lives under "Data & runs".
_NAV_KEYS = ["overview", "run", "compare", "feasibility", "data"]
_NAV_LABELS = ["Overview", "Run results", "Compare", "Feasibility", "Data & runs"]
_NAV_ICONS = ["house", "graph-up", "bar-chart-line", "speedometer2", "database"]
_PAGES = {
    "overview": home.render,
    "run": run_tab.render,
    "compare": comparison.render,
    "feasibility": feasibility.render,
    "data": data_models.render,
}

with st.sidebar:
    st.markdown("### Training Dashboard")

    # ── Navigation (icon menu, single level) ──────────────────────────────────
    if "nav" not in st.session_state:
        st.session_state["nav"] = "overview"
    # A hub card (or any page) can request a jump via "_nav_jump"; option_menu's
    # manual_select forces the menu to that entry on the next run.
    _manual = None
    if st.session_state.get("_nav_jump"):
        _manual = _NAV_KEYS.index(st.session_state.pop("_nav_jump"))
    _chosen = option_menu(
        menu_title=None, options=_NAV_LABELS, icons=_NAV_ICONS,
        default_index=_NAV_KEYS.index(st.session_state["nav"]),
        manual_select=_manual, key="navmenu",
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "nav-link": {"font-size": "0.9rem", "padding": "0.45rem 0.7rem",
                         "margin": "0.1rem 0"},
            "nav-link-selected": {"background-color": "#1A5276"},
        },
    )
    _page = _NAV_KEYS[_NAV_LABELS.index(_chosen)]
    st.session_state["nav"] = _page

    st.markdown("---")
    # ── Active run (shared across pages) ──────────────────────────────────────
    # Source of truth: st.session_state["run_label"]. It is set here OR by
    # clicking a row in the Overview "All runs" table (wandb-style selection).
    if not runs:
        st.warning("No runs found in logs/.")
        selected_run = None
        run = None
    else:
        _by_label = {r.label: r for r in runs}
        _active = st.session_state.get("run_label")
        if _active not in _by_label:
            _active = runs[0].label
        run = _by_label[_active]
        selected_run = run

        st.caption("**Active run**")
        st.markdown(f"<div class='active-run'>{_active}</div>", unsafe_allow_html=True)

        # Quick switch: filter by environment first (cuts the list to a few) so
        # each dropdown is short and readable. Browsing all runs at once is done
        # in the Overview table (full names, tags, sparklines).
        _envs = ["all"] + sorted({r.env for r in runs})
        _env_idx = _envs.index(run.env) if run.env in _envs else 0
        _env = st.selectbox("Environment", _envs, index=_env_idx, key="run_env_filter")
        _opts = [r.label for r in runs if _env == "all" or r.env == _env]
        if _active not in _opts:
            _opts = [_active] + _opts

        def _short(lbl: str) -> str:
            # Drop the date (all same-ish) and, when filtering by env, the [env]
            # tag, so the distinguishing part (time + model + tags) fits the box.
            import re
            s = re.sub(r"^\d{2}/\d{2}/\d{4}\s+", "", lbl)
            if _env != "all":
                s = s.replace(f"[{_env}] ", "")
            return s

        _picked = st.selectbox("Switch run", _opts, index=_opts.index(_active),
                               format_func=_short, key="run_switch")
        if _picked != _active:
            st.session_state["run_label"] = _picked
            st.rerun()
        st.caption("Browse all runs in the Overview table.")

    st.markdown("---")
    # ── Language (least-used control — keep it at the bottom) ─────────────────
    _choice = st.radio(
        "Language / Idioma", ["English", "Español"],
        index=0 if _lang == "en" else 1, horizontal=True, key="_lang_radio",
    )
    _new_lang = "es" if _choice == "Español" else "en"
    if _new_lang != _lang:
        st.session_state["_lang"] = _new_lang
        st.rerun()

# ── Build shared context and render the selected page ───────────────────────────
# (The refresh slider lives in System — Monitor/Live are the only consumers.)

ctx = DashboardContext(
    runs=runs,
    selected_run=selected_run,
    run=run,
    refresh_interval=10,
)
_PAGES.get(_page, home.render)(ctx)
