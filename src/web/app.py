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

from src.web.ui import i18n
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import _get_runs
from src.web.tabs import (
    home,
    run as run_tab,
    comparison,
    feasibility,
    data_models,
    system,
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
  [data-testid="stSidebar"] { min-width: 240px; max-width: 260px; }
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
  /* Selectbox dropdowns copy the control's width (~200px in the sidebar),
     truncating the run tags. Widening the popover itself breaks its centered
     positioning, so instead let the option list overflow it to the right. */
  [data-baseweb="popover"], [data-baseweb="popover"] > div,
  [data-baseweb="popover"] > div > div { overflow: visible !important; }
  [data-testid="stSelectboxVirtualDropdown"] {
    min-width: 27rem !important;
    background: white; box-shadow: 0 4px 16px rgba(0,0,0,0.16); border-radius: 0.5rem;
  }
  [data-testid="stSelectboxVirtualDropdown"] li { white-space: nowrap !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

runs = _get_runs()

# Grouped sidebar navigation: a single, always-visible map of the dashboard,
# organized by task. Each entry renders one tab module (full-width main area).
_NAV = [
    ("ANALYZE", [("overview", "Overview"), ("run", "Run results"), ("compare", "Compare")]),
    ("PLAN", [("feasibility", "Feasibility")]),
    ("DATA & OPS", [("data", "Data & models"), ("system", "System")]),
]
_PAGES = {
    "overview": home.render,
    "run": run_tab.render,
    "compare": comparison.render,
    "feasibility": feasibility.render,
    "data": data_models.render,
    "system": system.render,
}

# Everything must fit without scrolling: navigation first, then the run
# selector; the language toggle goes last (used once per session at most).
with st.sidebar:
    st.markdown("### Training Dashboard")

    # ── Navigation (grouped, always visible) ──────────────────────────────────
    _page = st.session_state.get("nav", "overview")
    for _group, _items in _NAV:
        st.caption(_group)
        for _key, _label in _items:
            if st.button(_label, key=f"nav_{_key}", use_container_width=True,
                         type="primary" if _page == _key else "secondary"):
                st.session_state["nav"] = _key
                st.rerun()
    _page = st.session_state.get("nav", "overview")

    st.markdown("---")
    # ── Run selector (shared context across pages) ────────────────────────────
    if not runs:
        st.warning("No runs found in logs/.")
        selected_run = None
        run = None
    else:
        trace_filter = st.selectbox("Trace mode", ["all", "simple", "deep"])
        filtered = [r for r in runs if trace_filter == "all" or r.trace_mode == trace_filter]

        if not filtered:
            st.warning("No runs match this filter.")
            selected_run = None
            run = None
        else:
            run_labels = {r.label: r for r in filtered}
            selected_label = st.selectbox("Run", list(run_labels.keys()))
            run = run_labels[selected_label]
            selected_run = run

            # The label already carries env/model/mode/precision — only add
            # what it doesn't: the log filename and which CSVs exist.
            has_csv = run.epoch_csv_path is not None and run.epoch_csv_path.exists()
            _csv_bits = " · ".join(
                f"{name} {'✓' if ok else '—'}"
                for name, ok in (("epoch", has_csv),
                                 ("batch", run.batch_csv_path is not None),
                                 ("per-class", run.perclass_csv_path is not None))
            )
            st.caption(f"{run.log_path.name}  \nCSV: {_csv_bits}")

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
