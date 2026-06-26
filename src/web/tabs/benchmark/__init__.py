"""Benchmark page — orchestrator. Three tabs: Predict / Compare vs runs / Report.
Predict is the analytic predictor (closed-form, no GPU); Compare puts predictions
next to real runs; Report reads a benchmark generated in the terminal
(`paravit benchmark`). The web never *trains* — Predict only computes formulas."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.web.benchmark_parser import parse_benchmark_csv
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import _get_benchmark_csvs, _bench_label
from src.web.tabs.benchmark.predict import _analytic_predictor
from src.web.tabs.benchmark.validate import render_validate, render_f1_prediction
from src.web.tabs.benchmark.report import render_report
from src.web.tabs.benchmark.study import render_study
from src.web.tabs.benchmark.ddp import render_ddp_analysis


def render(ctx: DashboardContext) -> None:
    selected_run = ctx.selected_run
    st.markdown("## Estimate / Benchmark")
    st.caption("**Estimate** computes any config from formulas — analytic, no GPU · "
               "**Benchmark** is a real (empirical) measurement generated in the "
               "terminal with `paravit benchmark` · **Benchmark vs Run** puts those "
               "estimates next to what actually happened in the real runs.")

    benchmark_csvs = _get_benchmark_csvs()
    tab_estimate, tab_benchmark, tab_compare = st.tabs(
        ["Estimate", "Benchmark", "Benchmark vs Run"])

    # ── Estimate: closed-form estimate for any config (no GPU, just formulas) ────
    with tab_estimate:
        _analytic_predictor()

    # The benchmark selector (feeds the Benchmark tab and Benchmark-vs-Run's F1 curve).
    if benchmark_csvs:
        with tab_benchmark:
            selected_feas_path = st.selectbox(
                "Benchmark report", [str(p) for p in benchmark_csvs],
                format_func=_bench_label, key="bench_sel",
                help="Used by the Benchmark tab and Benchmark-vs-Run's F1 curve.")
        meta, bdf_feas = parse_benchmark_csv(Path(selected_feas_path))
    else:
        meta, bdf_feas = {}, pd.DataFrame()

    # ── Benchmark: read a benchmark generated from the terminal (web only watches) ─
    with tab_benchmark:
        if not benchmark_csvs:
            st.info("No benchmarks yet. Generate one from the terminal — e.g. "
                    "`uv run paravit.py benchmark --model vit_base_patch16_224 "
                    "--batch-sizes 32,64` — and it will appear here.")
        else:
            st.caption("Benchmark selected above (generated in the terminal with "
                       "`paravit benchmark`): hardware, throughput, time and cost "
                       "estimates, distributed scaling and the convergence study.")
            subtab_ddp_opt = render_report(meta, bdf_feas, benchmark_csvs)
            with subtab_ddp_opt:
                render_ddp_analysis(meta, benchmark_csvs)
            st.markdown("---")
            render_study(meta, benchmark_csvs)

    # ── Benchmark vs Run: estimates next to the real run results ────────────────
    with tab_compare:
        subtab_prediction = render_validate(ctx)
        with subtab_prediction:
            with st.expander("Expected F1 curve (empirical prior vs the active run)"):
                render_f1_prediction(meta, selected_run, benchmark_csvs)
