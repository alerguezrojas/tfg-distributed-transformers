"""Tab render module — see src/web/app.py for the orchestrator."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.web.confusion_matrix_parser import get_matrix_for_epoch, parse_confusion_matrix_csv
from src.web.dataset_stats import (
    CLASS_NAMES, SPLIT_SIZES,
    class_distribution_approximate, class_distribution_from_parquet,
    get_country_distribution, find_example_patches, load_rgb_image,
)
from src.web.feasibility_comparison import build_comparison
from src.web.feasibility_parser import parse_feasibility_csv, parse_ddp_scenarios
from src.web.model_explorer import ALL_FAMILIES, CURATED_MODELS, compare_models
from src.web.perclass_parser import parse_perclass_csv
from src.web.run_registry import RunInfo
from src.web.system_monitor import get_snapshot

from src.web.ui.charts import (
    COLORS, _show, _dl_csv, _base_layout, _metric_fig, _overlay_fig,
    _CLASS_GROUPS, _CLASS_GROUP_COLOR,
)
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (
    ROOT, _load_df, _load_batch, _load_perclass, _get_runs, _get_feasibility_csvs,
    _feas_label, _run_config, _load_class_distribution, _load_example_images,
    _safe_max, _safe_idxmax, _safe_val_at_best, _throughput_col, _dur_str,
    _get_configs, _detect_anomalies, _read_log_tail, _parse_log_progress,
    _gpu_usage, _launch_process, _color_f1_cell,
)


# Sections reachable from the hub cards (key, title, one-line description).
_NAV_CARDS = [
    ("run", "Run results", "Curves, per-class, batch & metadata of one run"),
    ("compare", "Compare", "Speedup, energy & overlays across runs"),
    ("feasibility", "Feasibility", "Predict time, memory & cost before training"),
    ("data", "Data & models", "BigEarthNet explorer & timm models"),
    ("system", "System", "Hardware monitor & import remote runs"),
]


def render(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    st.markdown("## Overview")
    st.caption("A bit of everything in one place — global stats, the selected run, "
               "quick links and all runs. Drill into any section from the cards below.")

    # ── Global statistics (one pass over the runs) ──────────────────────────────
    best_f1_global = float("-inf")
    best_run_label = "—"
    best_run_df = pd.DataFrame()
    total_gpu_h = 0.0
    feasibility_csvs_home = _get_feasibility_csvs()
    curve_by_label: dict[str, list[float]] = {}

    for r in runs:
        try:
            df_r = _load_df(str(r.log_path),
                            str(r.epoch_csv_path) if r.epoch_csv_path else None)
            if not df_r.empty and "val_f1" in df_r.columns:
                curve_by_label[r.label] = df_r["val_f1"].dropna().round(4).tolist()
                run_best = _safe_max(df_r["val_f1"])
                if not pd.isna(run_best) and run_best > best_f1_global:
                    best_f1_global, best_run_label, best_run_df = run_best, r.label, df_r
            if not df_r.empty and "epoch_time" in df_r.columns:
                total_gpu_h += float(df_r["epoch_time"].dropna().sum()) / 3600
        except Exception:
            pass

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value in (
        (k1, "Total runs", str(len(runs))),
        (k2, "Best Val F1", f"{best_f1_global:.4f}" if best_f1_global > float("-inf") else "—"),
        (k3, "Total GPU time", f"{total_gpu_h:.1f} h"),
        (k4, "Feasibility reports", str(len(feasibility_csvs_home))),
    ):
        with col.container(border=True):
            st.caption(label)
            st.markdown(f"#### {value}")

    # ── Quick-nav cards (drill into each section) ───────────────────────────────
    st.markdown("#### Sections")
    nav_cols = st.columns(len(_NAV_CARDS))
    for col, (key, title, desc) in zip(nav_cols, _NAV_CARDS):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)
            if st.button("Open", key=f"hub_nav_{key}", use_container_width=True):
                st.session_state["nav"] = key
                st.rerun()

    # ── Highlight: the best run so far ──────────────────────────────────────────
    if not best_run_df.empty:
        st.markdown("#### Top run")
        with st.container(border=True):
            st.caption(best_run_label)
            _run_highlight(best_run_df)

    # ── Selected run at a glance ────────────────────────────────────────────────
    if selected_run is not None and selected_run.label != best_run_label:
        try:
            df_sel = _load_df(str(selected_run.log_path),
                              str(selected_run.epoch_csv_path) if selected_run.epoch_csv_path else None)
        except Exception:
            df_sel = pd.DataFrame()
        if not df_sel.empty and "val_f1" in df_sel.columns:
            st.markdown("#### Selected run")
            with st.container(border=True):
                st.caption(selected_run.label)
                _run_highlight(df_sel, anomalies_path=selected_run.log_path)

    # ── All runs with Val F1 sparklines ─────────────────────────────────────────
    st.markdown("#### All runs")
    _all_runs_table(runs, curve_by_label)


def _run_highlight(df: pd.DataFrame, anomalies_path=None) -> None:
    """Compact metric strip + two mini curves + a one-line verdict (card body)."""
    best_f1 = _safe_max(df["val_f1"])
    best_ep = _safe_val_at_best(df, "val_f1", "epoch")
    dur = (_dur_str(df["epoch_time"].dropna().sum())
           if "epoch_time" in df.columns and df["epoch_time"].notna().any() else "—")
    thr = (_safe_max(df["f1_at_threshold"])
           if "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any() else None)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Epochs", len(df))
    m2.metric("Best Val F1", f"{best_f1:.4f}" if not pd.isna(best_f1) else "—")
    m3.metric("Best epoch", int(best_ep) if best_ep is not None else "—")
    m4.metric("Duration", dur)
    if thr is not None:
        st.caption(f"F1 at the optimal threshold: {thr:.4f}")

    cc1, cc2 = st.columns(2)
    with cc1:
        fig = go.Figure()
        if "train_f1" in df.columns:
            fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_f1"], name="Train",
                                     line=dict(color=COLORS[0], width=2)))
        fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_f1"], name="Val",
                                 line=dict(color=COLORS[1], width=2)))
        fig.update_layout(**_base_layout(200, "F1 (macro)"), xaxis_title="Epoch", yaxis_title="F1")
        _show(fig, "hub_f1")
    with cc2:
        if "val_loss" in df.columns:
            fig = go.Figure()
            if "train_loss" in df.columns:
                fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train",
                                         line=dict(color=COLORS[0], width=2)))
            fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], name="Val",
                                     line=dict(color=COLORS[3], width=2)))
            fig.update_layout(**_base_layout(200, "Loss (BCE)"), xaxis_title="Epoch", yaxis_title="Loss")
            _show(fig, "hub_loss")

    # One-line verdict: overfitting gap at the best epoch.
    if "train_f1" in df.columns and best_ep is not None and not pd.isna(best_f1):
        _tr = df.loc[df["epoch"] == best_ep, "train_f1"]
        if not _tr.empty:
            gap = float(_tr.iloc[0]) - float(best_f1)
            note = " — overfitting" if gap > 0.1 else ""
            st.caption(f"Best Val F1 {best_f1:.3f} at epoch {int(best_ep)} · "
                       f"train–val gap {gap:+.2f}{note}")
    if anomalies_path is not None:
        anoms = _detect_anomalies(anomalies_path)
        (st.warning if anoms else st.success)(
            f"{len(anoms)} anomaly(ies) in the log" if anoms else "No anomalies detected")


def _all_runs_table(runs, curve_by_label: dict[str, list[float]]) -> None:
    """Runs table with a wandb-style Val F1 sparkline column."""
    rows = []
    for r in runs[:40]:
        try:
            df_r = _load_df(str(r.log_path),
                            str(r.epoch_csv_path) if r.epoch_csv_path else None)
            if df_r.empty or "val_f1" not in df_r.columns:
                continue
            best_f1 = _safe_max(df_r["val_f1"])
            if pd.isna(best_f1):
                continue
            best_ep = _safe_val_at_best(df_r, "val_f1", "epoch")
            dur_s = df_r["epoch_time"].dropna().sum() if "epoch_time" in df_r.columns else float("nan")
            energy_wh = (df_r["energy_eval_wh"].sum()
                         if "energy_eval_wh" in df_r.columns and df_r["energy_eval_wh"].notna().any()
                         else None)
            rows.append({
                "Run": r.label,
                "Val F1 curve": curve_by_label.get(r.label, []),
                "Mode": r.mode,
                "Precision": r.precision or "fp32",
                "Env": r.env,
                "Epochs": len(df_r),
                "Best Val F1": round(best_f1, 4),
                "Best epoch": int(best_ep) if best_ep is not None else None,
                "Duration": _dur_str(dur_s) if not pd.isna(dur_s) else "—",
                "Eval Wh": round(energy_wh) if energy_wh else None,
            })
        except Exception:
            pass

    if not rows:
        st.info("No runs with parseable metrics found.")
        return

    ov_df = pd.DataFrame(rows)
    _f1 = ov_df["Best Val F1"]
    st.dataframe(
        ov_df,
        use_container_width=True, hide_index=True,
        column_config={
            "Val F1 curve": st.column_config.LineChartColumn(
                "Val F1 curve", y_min=0.0, y_max=float(_f1.max()) + 0.05, width="medium"),
            "Best Val F1": st.column_config.ProgressColumn(
                "Best Val F1", min_value=0.0, max_value=float(_f1.max()) + 1e-9,
                format="%.4f"),
        },
    )
    _dl_csv(ov_df.drop(columns=["Val F1 curve"]), "runs_summary.csv", "Download runs table")

