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

from src.web.ui import i18n, theme
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import _get_runs
from src.web.tabs import (
    home,
    run as run_tab,
    comparison,
    analysis,
    feasibility,
    dataset,
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

# Design system: one global Plotly template + the typography/layout CSS.
theme.register_plotly_template()
theme.inject_css()


# ── Sidebar ────────────────────────────────────────────────────────────────────

runs = _get_runs()

# Single-level sidebar navigation (icon menu). Each entry renders one page
# module full-width. Pages keep at most ONE row of tabs inside — never 3 levels.
# System was removed: the live hardware monitor was not useful with the Kaggle
# workflow, and "Import runs" now lives under "Data & runs".
_NAV_KEYS = ["overview", "run", "compare", "analysis", "feasibility", "data"]
_NAV_LABELS = ["Overview", "Run results", "Compare", "Analysis", "Feasibility", "Dataset"]
_NAV_ICONS = ["house", "graph-up", "bar-chart-line", "diagram-3", "speedometer2", "database"]
_PAGES = {
    "overview": home.render,
    "run": run_tab.render,
    "compare": comparison.render,
    "analysis": analysis.render,
    "feasibility": feasibility.render,
    "data": dataset.render,
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
            "icon": {"display": "none"},
            "nav-link": {"font-size": "0.88rem", "font-weight": "500", "color": "#2B2F35",
                         "padding": "0.4rem 0.7rem", "margin": "0.05rem 0", "border-radius": "2px"},
            "nav-link-selected": {"background-color": "#EAEEF2", "color": "#1A1A1A",
                                  "font-weight": "600"},
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

        # Optional environment filter to narrow the list; the selector itself shows
        # the FULL distinguishing label (date · [env] · model · [mode] · [precision])
        # and is type-to-search, so runs are easy to tell apart and find.
        _envs = ["all environments"] + sorted({r.env for r in runs})
        _env = st.selectbox("Filter by environment", _envs, index=0, key="run_env_filter")
        _opts = [r.label for r in runs if _env == "all environments" or r.env == _env]
        if _active not in _opts:
            _opts = [_active] + _opts

        def _disp(lbl: str) -> str:
            # Front-load the distinguishing parts (model · tags · env) so runs are
            # told apart at a glance and the date is what truncates, not the tags.
            r = _by_label.get(lbl)
            if r is None:
                return lbl
            tags = []
            if r.mode != "single":
                tags.append(f"[{r.mode}]")
            if r.precision and r.precision != "fp32":
                tags.append(f"[{r.precision}]")
            if r.trace_mode == "deep":
                tags.append("[deep]")
            model = (r.model or "?").replace("_patch16_224", "")
            head = " ".join([model] + tags)
            return f"{head} · {r.env} · {lbl[:16]}"

        _picked = st.selectbox("Switch run", _opts, index=_opts.index(_active),
                               format_func=_disp, key="run_switch")
        if _picked != _active:
            st.session_state["run_label"] = _picked
            st.rerun()
        st.caption("Model · tags · env · date. Type to search, or click a row in Overview.")

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
