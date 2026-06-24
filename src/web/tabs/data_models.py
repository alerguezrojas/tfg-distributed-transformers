"""Import section — render module (see src/web/app.py for the orchestrator).

Trainings run on Kaggle or the cluster and produce a ``logs/`` folder. This page
imports those artifacts into the repo's ``logs/`` so the dashboard discovers
them. The dataset summary now lives in Overview; the timm model browser was
removed (the Benchmark predictor already covers model specs).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.web.run_import import import_run_archive, import_run_folder, summarize_import

from src.web.ui.context import DashboardContext
from src.web.ui.helpers import ROOT, _get_runs, _get_benchmark_csvs


def render(ctx: DashboardContext) -> None:
    st.markdown("## Import runs")
    st.caption("Bring in runs trained elsewhere (Kaggle, the cluster). The dataset "
               "summary is on the Overview page.")
    logs_root = ROOT / "logs"

    st.markdown("#### From a zip file")
    st.caption("A training elsewhere produces a `logs/` folder. Download it as a zip "
               "and drop it here — the artifacts are copied into `logs/` and appear "
               "in the dashboard immediately.")
    uploaded = st.file_uploader(
        "Drop a zip of the run's `logs/` folder (or its contents)",
        type=["zip"], accept_multiple_files=False,
    )
    if uploaded is not None and st.button("Import zip", type="primary"):
        try:
            rel = import_run_archive(uploaded.getvalue(), logs_root)
        except Exception as e:
            st.error(f"Could not read the zip: {e}")
            rel = []
        _report_import(rel)

    st.markdown("#### From a folder on this machine")
    st.caption("Useful when you already copied the `logs/` folder somewhere "
               "(for example via `scp` from the cluster).")
    folder_str = st.text_input("Folder path", placeholder="/home/alejandro/Downloads/kaggle_logs")
    if folder_str and st.button("Import folder"):
        folder = Path(folder_str).expanduser()
        if not folder.is_dir():
            st.error(f"Not a folder: {folder}")
        else:
            _report_import(import_run_folder(folder, logs_root))

    st.markdown("---")
    runs = ctx.runs
    envs = sorted({r.env for r in runs})
    c1, c2, c3 = st.columns(3)
    c1.metric("Runs indexed", len(runs))
    c2.metric("Benchmark reports", len(_get_benchmark_csvs()))
    c3.metric("Environments", ", ".join(envs) if envs else "—")


def _report_import(rel_paths: list[str]) -> None:
    if not rel_paths:
        st.warning("No recognizable artifacts found "
                   "(expected train_*.log / *_metrics_*.csv / confusion_matrix_*.csv / "
                   "benchmark_*).")
        return
    s = summarize_import(rel_paths)
    _get_runs.clear()
    _get_benchmark_csvs.clear()
    st.success(
        f"Imported {s['total']} file(s): {s['runs']} run log(s), "
        f"{s['metric_csvs']} metric CSV(s), {s['benchmark']} benchmark report(s)."
    )
    with st.expander("Imported files"):
        for p in rel_paths:
            st.write(f"`logs/{p}`")
    st.info("Select the new run in the sidebar to explore it.")
