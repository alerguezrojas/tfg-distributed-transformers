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


def render(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    st.markdown("## Project overview")

    # ── Global statistics ──────────────────────────────────────────────────────
    total_runs = len(runs)
    best_f1_global = float("-inf")
    best_run_label = "—"
    total_gpu_h = 0.0
    feasibility_csvs_home = _get_feasibility_csvs()

    for r in runs:
        try:
            df_r = _load_df(
                str(r.log_path),
                str(r.epoch_csv_path) if r.epoch_csv_path else None,
            )
            if not df_r.empty and "val_f1" in df_r.columns:
                run_best = _safe_max(df_r["val_f1"])
                if not pd.isna(run_best) and run_best > best_f1_global:
                    best_f1_global = run_best
                    best_run_label = r.label
            if not df_r.empty and "epoch_time" in df_r.columns:
                total_gpu_h += float(df_r["epoch_time"].dropna().sum()) / 3600
        except Exception:
            pass

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Total runs", total_runs)
    g2.metric("Best Val F1", f"{best_f1_global:.4f}" if best_f1_global > float("-inf") else "—")
    g3.metric("Top run", best_run_label[:28] if best_run_label != "—" else "—")
    g4.metric("Total GPU time", f"{total_gpu_h:.1f} h")
    g5.metric("Feasibility reports", len(feasibility_csvs_home))

    st.markdown("---")

    # ── Selected run: summary + mini curves ─────────────────────────────────────
    if selected_run is not None:
        st.markdown(f"### Selected run — `{selected_run.label}`")
        try:
            df_sel = _load_df(
                str(selected_run.log_path),
                str(selected_run.epoch_csv_path) if selected_run.epoch_csv_path else None,
            )
        except Exception:
            df_sel = pd.DataFrame()

        col_meta, col_curves = st.columns([1, 2])

        with col_meta:
            if not df_sel.empty and "val_f1" in df_sel.columns:
                best_f1_sel = _safe_max(df_sel["val_f1"])
                best_ep_v = _safe_val_at_best(df_sel, "val_f1", "epoch")
                n_ep_sel = len(df_sel)
                dur_sel = ""
                if "epoch_time" in df_sel.columns and df_sel["epoch_time"].notna().any():
                    dur_sel = _dur_str(df_sel["epoch_time"].dropna().sum())
                thresh_f1 = (
                    _safe_max(df_sel["f1_at_threshold"])
                    if "f1_at_threshold" in df_sel.columns and df_sel["f1_at_threshold"].notna().any()
                    else None
                )
                m1, m2 = st.columns(2)
                m1.metric("Epochs completed", n_ep_sel)
                m2.metric("Best Val F1", f"{best_f1_sel:.4f}" if not pd.isna(best_f1_sel) else "—")
                m3, m4 = st.columns(2)
                m3.metric("Best epoch", int(best_ep_v) if best_ep_v is not None else "—")
                m4.metric("Duration", dur_sel or "—")
                if thresh_f1 is not None:
                    st.metric("F1 @ optimal threshold", f"{thresh_f1:.4f}")
                # One-line verdict: overfitting gap at the best epoch.
                if "train_f1" in df_sel.columns and best_ep_v is not None and not pd.isna(best_f1_sel):
                    _tr = df_sel.loc[df_sel["epoch"] == best_ep_v, "train_f1"]
                    if not _tr.empty:
                        gap = float(_tr.iloc[0]) - float(best_f1_sel)
                        note = " — overfitting" if gap > 0.1 else ""
                        st.caption(f"Best Val F1 {best_f1_sel:.3f} at epoch "
                                   f"{int(best_ep_v)} · train–val gap {gap:+.2f}{note}")
                anomalies_home = _detect_anomalies(selected_run.log_path)
                if anomalies_home:
                    st.warning(f"{len(anomalies_home)} anomaly(ies) in the log")
                else:
                    st.success("No anomalies detected")
            else:
                st.info("No metrics data for this run.")

        with col_curves:
            if not df_sel.empty and "val_f1" in df_sel.columns:
                cc1, cc2 = st.columns(2)
                with cc1:
                    fig_f1_home = go.Figure()
                    if "train_f1" in df_sel.columns:
                        fig_f1_home.add_trace(go.Scatter(
                            x=df_sel["epoch"], y=df_sel["train_f1"],
                            name="Train", line=dict(color=COLORS[0], width=2),
                        ))
                    fig_f1_home.add_trace(go.Scatter(
                        x=df_sel["epoch"], y=df_sel["val_f1"],
                        name="Val", line=dict(color=COLORS[1], width=2),
                    ))
                    fig_f1_home.update_layout(
                        **_base_layout(200, "F1 (macro)"),
                        xaxis_title="Epoch", yaxis_title="F1",
                    )
                    _show(fig_f1_home, "inicio_f1")
                with cc2:
                    if "val_loss" in df_sel.columns:
                        fig_loss_home = go.Figure()
                        if "train_loss" in df_sel.columns:
                            fig_loss_home.add_trace(go.Scatter(
                                x=df_sel["epoch"], y=df_sel["train_loss"],
                                name="Train", line=dict(color=COLORS[0], width=2),
                            ))
                        fig_loss_home.add_trace(go.Scatter(
                            x=df_sel["epoch"], y=df_sel["val_loss"],
                            name="Val", line=dict(color=COLORS[3], width=2),
                        ))
                        fig_loss_home.update_layout(
                            **_base_layout(200, "Loss (BCE)"),
                            xaxis_title="Epoch", yaxis_title="Loss",
                        )
                        _show(fig_loss_home, "inicio_loss")

    st.markdown("---")
    st.caption(
        "Tip: use the sidebar menu to navigate. Live system info lives in **System → Monitor**; "
        "per-class results in **Run results → Per-class**."
    )

    # ── Table of all runs ───────────────────────────────────────────────────────
    st.markdown("### All runs")
    overview_rows = []
    for r in runs[:30]:
        try:
            df_r = _load_df(
                str(r.log_path),
                str(r.epoch_csv_path) if r.epoch_csv_path else None,
            )
            if df_r.empty or "val_f1" not in df_r.columns:
                continue
            run_best_f1 = _safe_max(df_r["val_f1"])
            if pd.isna(run_best_f1):
                continue
            best_ep_v = _safe_val_at_best(df_r, "val_f1", "epoch")
            dur_s = df_r["epoch_time"].dropna().sum() if "epoch_time" in df_r.columns else float("nan")
            energy_wh = (
                df_r["energy_eval_wh"].sum()
                if "energy_eval_wh" in df_r.columns and df_r["energy_eval_wh"].notna().any()
                else None
            )
            overview_rows.append({
                "Run": r.label[:55],
                "Environment": r.env,
                "Model": r.model or "—",
                "Trace": r.trace_mode,
                "Epochs": len(df_r),
                "Best Val F1": round(run_best_f1, 4),
                "Best epoch": int(best_ep_v) if best_ep_v is not None else "—",
                "Duration": _dur_str(dur_s) if not pd.isna(dur_s) else "—",
                "Eval energy (Wh)": f"{energy_wh:.0f}" if energy_wh else "—",
            })
        except Exception:
            pass

    if overview_rows:
        ov_df = pd.DataFrame(overview_rows)
        st.dataframe(
            ov_df.style.background_gradient(subset=["Best Val F1"], cmap="RdYlGn", vmin=0.4, vmax=0.75),
            use_container_width=True, hide_index=True,
        )
        _dl_csv(ov_df, "runs_summary.csv", "Download runs table")
    else:
        st.info("No runs with parseable metrics found.")

