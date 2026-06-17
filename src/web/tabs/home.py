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
    ("data", "Data & runs", "Dataset explorer, models & import runs"),
]


def render(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    st.markdown("## Overview")
    st.caption("Project at a glance. Click a row in the table to make a run active; "
               "open any section from the cards.")

    # ── One pass over the runs (stats + per-run val_f1 curve) ───────────────────
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

    # ── Compact dashboard grid (KPIs · top run · best-F1 ranking · sections) ────
    # Row 1: KPIs (left) + the selected run's curves (right) — dense, no scroll.
    left, right = st.columns([1, 1.4])

    with left:
        k1, k2 = st.columns(2)
        _kpi(k1, "Runs", str(len(runs)))
        _kpi(k2, "Best Val F1", f"{best_f1_global:.3f}" if best_f1_global > float("-inf") else "—")
        k3, k4 = st.columns(2)
        _kpi(k3, "GPU time", f"{total_gpu_h:.0f} h")
        _kpi(k4, "Feasibility", str(len(feasibility_csvs_home)))
        with st.container(border=True):
            st.caption("Best Val F1 by run (top 8)")
            _best_f1_bars(runs, curve_by_label)

    with right:
        with st.container(border=True):
            _df_active = best_run_df
            _title = best_run_label
            if selected_run is not None and selected_run.label in curve_by_label:
                try:
                    _df_active = _load_df(
                        str(selected_run.log_path),
                        str(selected_run.epoch_csv_path) if selected_run.epoch_csv_path else None)
                    _title = selected_run.label
                except Exception:
                    pass
            st.caption(f"Active run — {_title}")
            if not _df_active.empty and "val_f1" in _df_active.columns:
                _run_highlight(_df_active)
            else:
                st.info("No metrics for this run.")

    # ── Section cards (compact, one row) ────────────────────────────────────────
    nav_cols = st.columns(len(_NAV_CARDS))
    for col, (key, title, desc) in zip(nav_cols, _NAV_CARDS):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)
            if st.button("Open", key=f"hub_nav_{key}", use_container_width=True):
                st.session_state["_nav_jump"] = key
                st.rerun()

    # ── All runs — selectable table (click a row → active run) ──────────────────
    st.markdown("#### All runs")
    st.caption("Click a row to make that run active across the dashboard.")
    _all_runs_table(runs, curve_by_label)


def _kpi(col, label: str, value: str) -> None:
    with col.container(border=True):
        st.caption(label)
        st.markdown(f"#### {value}")


def _best_f1_bars(runs, curve_by_label: dict[str, list[float]]) -> None:
    """Compact horizontal bar of the best Val F1 of the top-8 runs."""
    rows = []
    for r in runs:
        curve = curve_by_label.get(r.label)
        if curve:
            rows.append((r.label, max(curve)))
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:8]
    if not rows:
        st.info("No runs with metrics.")
        return
    labels = [l for l, _ in rows][::-1]
    vals = [v for _, v in rows][::-1]
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h", marker_color=COLORS[0],
        text=[f"{v:.3f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        height=40 + 30 * len(rows), margin=dict(l=10, r=30, t=6, b=6),
        paper_bgcolor="white", plot_bgcolor="#f8fafc", showlegend=False,
    )
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(automargin=True, tickfont=dict(size=9))
    _show(fig, "hub_bestf1_bars")


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
        fig.update_layout(**_base_layout(170, "F1 (macro)"), xaxis_title="Epoch", yaxis_title="F1")
        _show(fig, "hub_f1")
    with cc2:
        if "val_loss" in df.columns:
            fig = go.Figure()
            if "train_loss" in df.columns:
                fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train",
                                         line=dict(color=COLORS[0], width=2)))
            fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], name="Val",
                                     line=dict(color=COLORS[3], width=2)))
            fig.update_layout(**_base_layout(170, "Loss (BCE)"), xaxis_title="Epoch", yaxis_title="Loss")
            _show(fig, "hub_loss")

    # One-line verdict: overfitting gap at the best epoch.
    if "train_f1" in df.columns and best_ep is not None and not pd.isna(best_f1):
        _tr = df.loc[df["epoch"] == best_ep, "train_f1"]
        if not _tr.empty:
            gap = float(_tr.iloc[0]) - float(best_f1)
            note = " — overfitting" if gap > 0.1 else ""
            st.caption(f"Best Val F1 {best_f1:.3f} at epoch {int(best_ep)} · "
                       f"train–val gap {gap:+.2f}{note}")


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
    event = st.dataframe(
        ov_df,
        use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="runs_table",
        column_config={
            "Val F1 curve": st.column_config.LineChartColumn(
                "Val F1 curve", y_min=0.0, y_max=float(_f1.max()) + 0.05, width="medium"),
            "Best Val F1": st.column_config.ProgressColumn(
                "Best Val F1", min_value=0.0, max_value=float(_f1.max()) + 1e-9,
                format="%.4f"),
        },
    )
    # Clicking a row makes that run active across the whole dashboard. We act
    # only when the selected ROW changes (tracked in _last_table_row); otherwise
    # a stale table selection would override a run picked from the sidebar.
    sel = event.selection.rows if event and event.selection else []
    if sel:
        chosen = ov_df.iloc[sel[0]]["Run"]
        if st.session_state.get("_last_table_row") != chosen:
            st.session_state["_last_table_row"] = chosen
            st.session_state["run_label"] = chosen
            st.rerun()
    _dl_csv(ov_df.drop(columns=["Val F1 curve"]), "runs_summary.csv", "Download runs table")

