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
    st.caption("Plan before training. **Predict** with the analytic model (no run "
               "needed), **Validate** predictions against real trainings, or "
               "**Measure** on this machine to calibrate.")
    tab_predict, tab_validate, tab_measure = st.tabs(
        ["Predict", "Validate", "Measure (advanced)"])

    with tab_predict:
        _analytic_predictor()

    with tab_measure:
        st.caption("Run the real benchmark on the machine you are on. Use it to "
                   "calibrate the predictor or to profile this hardware.")
        st.markdown("#### Generate a report")
        subtab_run_feas = st.container()
        st.markdown("#### Report")
        subtab_report = st.container()
        st.markdown("#### Convergence study")
        subtab_study = st.container()

    feasibility_csvs = _get_feasibility_csvs()
    if feasibility_csvs:
        selected_feas_path = st.sidebar.selectbox(
            "Feasibility report", [str(p) for p in feasibility_csvs],
            format_func=_feas_label, key="feas_sidebar_sel")
        meta, bdf_feas = parse_feasibility_csv(Path(selected_feas_path))
    else:
        meta, bdf_feas = {}, pd.DataFrame()

    with tab_validate:
        subtab_prediction = render_validate(ctx)
    with subtab_report:
        subtab_ddp_opt = render_report(meta, bdf_feas, feasibility_csvs)
    with subtab_study:
        render_study(meta, feasibility_csvs)
    with subtab_ddp_opt:
        render_ddp_analysis(meta, feasibility_csvs)
    with subtab_prediction:
        render_f1_prediction(meta, selected_run, feasibility_csvs)
    with subtab_run_feas:
        render_run_form()
