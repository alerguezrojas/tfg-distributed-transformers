"""Feasibility page — orchestrator. Predict / Validate / Measure, one module
per responsibility (predict, validate, report, study, ddp, run_form)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.web.feasibility_parser import parse_feasibility_csv
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import _get_feasibility_csvs, _feas_label
from src.web.tabs.feasibility.predict import _analytic_predictor
from src.web.tabs.feasibility.validate import render_validate, render_f1_prediction
from src.web.tabs.feasibility.report import render_report
from src.web.tabs.feasibility.study import render_study
from src.web.tabs.feasibility.ddp import render_ddp_analysis
from src.web.tabs.feasibility.run_form import render_run_form


def render(ctx: DashboardContext) -> None:
    selected_run = ctx.selected_run
    st.markdown("## Feasibility")
    st.caption("**Compare vs runs** puts the predictions next to what actually "
               "happened · **Predict** estimates any config with the analytic model "
               "(no run needed) · **Measure (advanced)** benchmarks this machine.")

    # One visible report selector (it used to be hidden in the sidebar). It feeds
    # the F1 prediction in Compare and the whole Measure tab.
    feasibility_csvs = _get_feasibility_csvs()
    if feasibility_csvs:
        selected_feas_path = st.selectbox(
            "Feasibility report", [str(p) for p in feasibility_csvs],
            format_func=_feas_label, key="feas_sel",
            help="Used by Compare's F1 prediction and the Measure tab.")
        meta, bdf_feas = parse_feasibility_csv(Path(selected_feas_path))
    else:
        meta, bdf_feas = {}, pd.DataFrame()

    tab_compare, tab_predict, tab_measure = st.tabs(
        ["Compare vs runs", "Predict", "Measure (advanced)"])

    # ── Compare vs runs: predictions/estimates next to the real results ─────────
    with tab_compare:
        subtab_prediction = render_validate(ctx)
        with subtab_prediction:
            render_f1_prediction(meta, selected_run, feasibility_csvs)

    # ── Predict: analytic estimate for any config, no run needed ────────────────
    with tab_predict:
        _analytic_predictor()

    # ── Measure (advanced): real benchmark on this machine + study ──────────────
    with tab_measure:
        st.caption("Run the real benchmark on this machine to calibrate the "
                   "predictor or profile the hardware. The report shown is the one "
                   "selected above.")
        render_run_form()
        st.markdown("---")
        subtab_ddp_opt = render_report(meta, bdf_feas, feasibility_csvs)
        with subtab_ddp_opt:
            render_ddp_analysis(meta, feasibility_csvs)
        st.markdown("---")
        render_study(meta, feasibility_csvs)
