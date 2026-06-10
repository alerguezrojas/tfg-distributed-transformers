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
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

runs = _get_runs()

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

            st.markdown("---")
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
    st.markdown("**Live monitor**")
    refresh_interval = st.slider("Refresh interval (s)", 5, 60, 10)

# ── Build shared context and dispatch tabs ──────────────────────────────────────

ctx = DashboardContext(
    runs=runs,
    selected_run=selected_run,
    run=run,
    refresh_interval=refresh_interval,
)

tab_home, tab_run, tab_comp, tab_viability, tab_data, tab_system = st.tabs(
    ["Home", "Run", "Comparison", "Feasibility", "Data & models", "System"]
)
with tab_home:
    home.render(ctx)
with tab_run:
    run_tab.render(ctx)
with tab_comp:
    comparison.render(ctx)
with tab_viability:
    feasibility.render(ctx)
with tab_data:
    data_models.render(ctx)
with tab_system:
    system.render(ctx)
