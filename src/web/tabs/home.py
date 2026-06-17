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
    _feas_label, _run_config, _load_class_distribution, _load_example_images, _class_gallery,
    _safe_max, _safe_idxmax, _safe_val_at_best, _throughput_col, _dur_str,
    _get_configs, _detect_anomalies, _read_log_tail, _parse_log_progress,
    _gpu_usage, _launch_process, _color_f1_cell,
)


def render(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    st.markdown("## Overview")
    st.caption("Project at a glance. Click a row in the table to make a run active.")

    # ── One pass over the runs (stats + per-run val_f1 curve + epoch time) ──────
    best_f1_global = float("-inf")
    best_run_label = "—"
    best_run_df = pd.DataFrame()
    total_gpu_h = 0.0
    fastest_min = float("inf")
    total_energy_wh = 0.0
    feasibility_csvs_home = _get_feasibility_csvs()
    curve_by_label: dict[str, list[float]] = {}
    time_by_label: dict[str, float] = {}   # avg epoch time (s)
    mode_counts: dict[str, int] = {}

    for r in runs:
        mode_counts[r.mode] = mode_counts.get(r.mode, 0) + 1
        try:
            df_r = _load_df(str(r.log_path),
                            str(r.epoch_csv_path) if r.epoch_csv_path else None)
            if not df_r.empty and "val_f1" in df_r.columns:
                curve_by_label[r.label] = df_r["val_f1"].dropna().round(4).tolist()
                run_best = _safe_max(df_r["val_f1"])
                if not pd.isna(run_best) and run_best > best_f1_global:
                    best_f1_global, best_run_label, best_run_df = run_best, r.label, df_r
            if not df_r.empty and "epoch_time" in df_r.columns and df_r["epoch_time"].notna().any():
                total_gpu_h += float(df_r["epoch_time"].dropna().sum()) / 3600
                avg_s = float(df_r["epoch_time"].dropna().mean())
                time_by_label[r.label] = avg_s
                fastest_min = min(fastest_min, avg_s / 60)
            for _c in ("energy_train_wh", "energy_eval_wh"):
                if _c in df_r.columns and df_r[_c].notna().any():
                    total_energy_wh += float(df_r[_c].dropna().sum())
        except Exception:
            pass

    n_models = len({r.model for r in runs if r.model})
    n_envs = len({r.env for r in runs})

    # ── Compact KPI strip (one dense row) ───────────────────────────────────────
    _kpi_strip([
        ("Runs", str(len(runs))),
        ("Best Val F1", f"{best_f1_global:.3f}" if best_f1_global > float("-inf") else "—"),
        ("Fastest epoch", f"{fastest_min:.1f} min" if fastest_min < float("inf") else "—"),
        ("GPU time", f"{total_gpu_h:.0f} h"),
        ("Energy", f"{total_energy_wh:.0f} Wh" if total_energy_wh else "—"),
        ("Models", str(n_models)),
        ("Environments", str(n_envs)),
        ("Feasibility", str(len(feasibility_csvs_home))),
    ])

    # ── Row 1: best-F1 ranking (left) + active-run curves (right) ───────────────
    left, right = st.columns([1, 1.4])
    with left:
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

    # ── Row 2: relevant charts (where the nav cards used to be) ─────────────────
    g_left, g_right = st.columns([1.4, 1])
    with g_left:
        with st.container(border=True):
            st.caption("Training speed — average time per epoch (fastest 8, min)")
            _epoch_time_bars(time_by_label)
    with g_right:
        with st.container(border=True):
            st.caption("Runs by strategy")
            _strategy_donut(mode_counts)

    # ── Dataset at a glance (moved here from the old Dataset section) ───────────
    _dataset_panel()

    # ── All runs — selectable table (click a row → active run) ──────────────────
    st.markdown("#### All runs")
    st.caption("Click a row to make that run active across the dashboard.")
    _all_runs_table(runs, curve_by_label)


_META_CANDIDATES = [
    "/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet",
    "/home/bejeque/alu0101317038/datasets/bigearthnet/metadata.parquet",
]
_ROOT_CANDIDATES = [
    "/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2",
    "/home/bejeque/alu0101317038/datasets/bigearthnet/BigEarthNet-S2",
]


def _dataset_panel() -> None:
    """Compact dataset summary: splits + top classes + a few example patches."""
    meta = next((p for p in _META_CANDIDATES if Path(p).exists()), None)
    root = next((p for p in _ROOT_CANDIDATES if Path(p).exists()), None)

    st.markdown("#### Dataset — BigEarthNet-S2")
    d_left, d_right = st.columns([1, 1.1])
    with d_left:
        with st.container(border=True):
            s1, s2, s3 = st.columns(3)
            s1.metric("Train", f"{SPLIT_SIZES['train']:,}")
            s2.metric("Val", f"{SPLIT_SIZES['val']:,}")
            s3.metric("Test", f"{SPLIT_SIZES['test']:,}")
            st.caption(f"{sum(SPLIT_SIZES.values()):,} patches · 19 CORINE classes · "
                       "multi-label · RGB proxy (B04/B03/B02)")
    with d_right:
        with st.container(border=True):
            dist = _load_class_distribution(str(meta)) if meta else None
            if dist is None:
                dist = class_distribution_approximate()
            top = dist.sort_values("train_count").tail(8)
            st.caption("Most frequent classes (train patches)")
            fig = go.Figure(go.Bar(
                y=top["class"], x=top["train_count"], orientation="h",
                marker_color=COLORS[2]))
            fig.update_layout(
                height=190, margin=dict(l=10, r=10, t=6, b=6),
                paper_bgcolor="white", plot_bgcolor="#f8fafc", showlegend=False)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(automargin=True, tickfont=dict(size=9))
            _show(fig, "hub_dataset_dist")

    # ── Gallery: one example patch per class, captioned with its statistics ─────
    gallery = _class_gallery(str(meta), str(root)) if (meta and root) else []
    if gallery:
        st.caption("One example patch per class — caption shows train patches and "
                   "the share of training images that contain the class.")
        # Uniform column width (use_container_width) → even gaps; a fixed-height
        # caption box keeps the rows aligned despite different name lengths.
        ncols = 10
        for i in range(0, len(gallery), ncols):
            cols = st.columns(ncols, gap="small")
            for col, (cls, cnt, pct, img) in zip(cols, gallery[i:i + ncols]):
                col.image(img, use_container_width=True)
                col.markdown(
                    "<div style='font-size:0.66rem;line-height:1.1;height:3.0rem;"
                    f"overflow:hidden'><b>{cls}</b><br>{cnt:,} · {pct:.0f}%</div>",
                    unsafe_allow_html=True)
    elif not (meta and root):
        st.caption("Dataset not mounted on this machine — splits and class counts "
                   "shown from metadata.")


def _epoch_time_bars(time_by_label: dict[str, float]) -> None:
    """Fastest 8 runs by average epoch time (min) — the speed at a glance."""
    rows = sorted(time_by_label.items(), key=lambda x: x[1])[:8]
    if not rows:
        st.info("No runs with timing data.")
        return
    labels = [l for l, _ in rows][::-1]
    mins = [v / 60 for _, v in rows][::-1]
    fig = go.Figure(go.Bar(
        y=labels, x=mins, orientation="h", marker_color=COLORS[2],
        text=[f"{m:.1f}" for m in mins], textposition="outside",
    ))
    fig.update_layout(
        height=40 + 30 * len(rows), margin=dict(l=10, r=30, t=6, b=6),
        paper_bgcolor="white", plot_bgcolor="#f8fafc", showlegend=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(automargin=True, tickfont=dict(size=9))
    _show(fig, "hub_epoch_time")


def _strategy_donut(mode_counts: dict[str, int]) -> None:
    """How many runs of each strategy (single / ddp / model_parallel / hetero)."""
    if not mode_counts:
        st.info("No runs.")
        return
    _names = {"single": "Single-GPU", "ddp": "DDP", "model_parallel": "Model-parallel",
              "ddp_hetero": "Heterogeneous"}
    labels = [_names.get(k, k) for k in mode_counts]
    fig = go.Figure(go.Pie(
        labels=labels, values=list(mode_counts.values()), hole=0.55,
        marker=dict(colors=COLORS), textinfo="value",
    ))
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="top", y=-0.05, font=dict(size=10)),
    )
    _show(fig, "hub_strategy_donut")


def _kpi_strip(items: list[tuple[str, str]]) -> None:
    """One dense row of stat cards (much more compact than st.metric cards)."""
    cells = "".join(
        f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>"
        for l, v in items
    )
    st.markdown(f"<div class='kpi-strip'>{cells}</div>", unsafe_allow_html=True)


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

