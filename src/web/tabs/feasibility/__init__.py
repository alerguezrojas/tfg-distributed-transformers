"""Feasibility page — orchestrator. Two read-only tabs: Compare vs runs / Report
(validate, report, study, ddp). The analytic predictor lives in the terminal
(`tfg predict`); the web only *reads* — reports are generated with `tfg feasibility`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.web.feasibility_parser import parse_feasibility_csv
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import _get_feasibility_csvs, _feas_label
from src.web.tabs.feasibility.validate import render_validate, render_f1_prediction
from src.web.tabs.feasibility.report import render_report
from src.web.tabs.feasibility.study import render_study
from src.web.tabs.feasibility.ddp import render_ddp_analysis


def render(ctx: DashboardContext) -> None:
    selected_run = ctx.selected_run
    st.markdown("## Feasibility")
    st.caption("**Compare vs runs** puts the predictions next to what actually "
               "happened · **Report** shows a benchmark generated in the terminal. "
               "Plan a config without running with `tfg predict` in the terminal.")

    # One visible report selector. It feeds the F1 prediction in Compare and Report.
    feasibility_csvs = _get_feasibility_csvs()
    if feasibility_csvs:
        selected_feas_path = st.selectbox(
            "Feasibility report", [str(p) for p in feasibility_csvs],
            format_func=_feas_label, key="feas_sel",
            help="Used by Compare's F1 prediction and the Report tab.")
        meta, bdf_feas = parse_feasibility_csv(Path(selected_feas_path))
    else:
        meta, bdf_feas = {}, pd.DataFrame()

    tab_compare, tab_report = st.tabs(["Compare vs runs", "Report"])

    # ── Compare vs runs: predictions/estimates next to the real results ─────────
    with tab_compare:
        subtab_prediction = render_validate(ctx)
        with subtab_prediction:
            with st.expander("Expected F1 curve (empirical prior vs the active run)"):
                render_f1_prediction(meta, selected_run, feasibility_csvs)

    # ── Report: read a benchmark generated from the terminal (web only watches) ──
    with tab_report:
        if not feasibility_csvs:
            st.info("No feasibility reports yet. Generate one from the terminal — e.g. "
                    "`uv run tfg.py feasibility --model vit_base_patch16_224 "
                    "--batch-sizes 32,64` — and it will appear here.")
        else:
            st.caption("Benchmark of the report selected above (generated in the "
                       "terminal with `tfg feasibility`): hardware, throughput, time "
                       "and cost estimates, distributed scaling and the convergence study.")
            subtab_ddp_opt = render_report(meta, bdf_feas, feasibility_csvs)
            with subtab_ddp_opt:
                render_ddp_analysis(meta, feasibility_csvs)
            st.markdown("---")
            render_study(meta, feasibility_csvs)
