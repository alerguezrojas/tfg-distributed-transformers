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
  .block-container { padding-top: 4rem; padding-left: 1.5rem; padding-right: 1.5rem; }
  h1 { font-size: 1.4rem; font-weight: 600; }
  h2 { font-size: 1.1rem; font-weight: 600; margin-top: 1.2rem; }
  h3 { font-size: 0.95rem; font-weight: 600; }
  [data-testid="stMetricValue"] { font-size: 1.1rem; }
  [data-baseweb="tab-list"] {
    overflow-x: auto !important; flex-wrap: nowrap !important;
    scrollbar-width: thin; gap: 0 !important;
  }
  [data-baseweb="tab-list"]::-webkit-scrollbar { height: 3px; }
  [data-baseweb="tab-list"]::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
  [data-baseweb="tab"] {
    white-space: nowrap !important; font-size: 0.82rem !important;
    padding-left: 0.75rem !important; padding-right: 0.75rem !important;
    min-width: unset !important;
  }
  /* Sidebar navigation buttons: left-aligned, menu-like */
  [data-testid="stSidebar"] .stButton button {
    justify-content: flex-start; text-align: left; font-weight: 500;
    padding-top: 0.3rem; padding-bottom: 0.3rem;
  }
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    margin-top: 0.4rem; letter-spacing: 0.04em; opacity: 0.7;
  }
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

with st.sidebar:
    st.markdown("### Training Dashboard")
    _choice = st.radio(
        "Language / Idioma", ["English", "Español"],
        index=0 if _lang == "en" else 1, horizontal=True, key="_lang_radio",
    )
    _new_lang = "es" if _choice == "Español" else "en"
    if _new_lang != _lang:
        st.session_state["_lang"] = _new_lang
        st.rerun()

    st.markdown("---")
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

            has_csv = run.epoch_csv_path is not None and run.epoch_csv_path.exists()
            st.caption(
                f"**Log:** {run.log_path.name}  \n"
                f"**Environment:** {run.env}  \n"
                f"**Mode:** {run.mode}  \n"
                f"**Model:** {run.model or '—'}  \n"
                f"**Trace:** {run.trace_mode}  \n"
                f"**Epoch CSV:** {'yes' if has_csv else 'no'}  \n"
                f"**Batch CSV:** {'yes' if run.batch_csv_path else 'no'}  \n"
                f"**Per-class CSV:** {'yes' if run.perclass_csv_path else 'no'}"
            )

    st.markdown("---")
    refresh_interval = st.slider("Refresh interval (s)", 5, 60, 10)

# ── Build shared context and render the selected page ───────────────────────────

ctx = DashboardContext(
    runs=runs,
    selected_run=selected_run,
    run=run,
    refresh_interval=refresh_interval,
)
_PAGES.get(_page, home.render)(ctx)
