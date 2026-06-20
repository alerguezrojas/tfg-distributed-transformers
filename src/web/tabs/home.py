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
    _gpu_usage, _color_f1_cell,
)

# Where the BigEarthNet-S2 metadata / patches live (local SSD or Verode NFS). If
# neither is mounted, the charts fall back to an approximate class distribution
# and the photo strip is skipped.
_META_CANDIDATES = [
    "/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet",
    "/home/bejeque/alu0101317038/datasets/bigearthnet/metadata.parquet",
]
_ROOT_CANDIDATES = [
    "/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2",
    "/home/bejeque/alu0101317038/datasets/bigearthnet/BigEarthNet-S2",
]


def render(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    st.markdown("## Overview")

    # ── One pass over the runs (stats + per-run val_f1 curve + epoch time) ──────
    best_f1_global = float("-inf")
    best_run_label = "—"
    best_run_df = pd.DataFrame()
    total_gpu_h = 0.0
    fastest_min = float("inf")
    total_energy_wh = 0.0
    feasibility_csvs_home = _get_feasibility_csvs()
    curve_by_label: dict[str, list[float]] = {}
    gpu_secs_by_env: dict[str, float] = {}     # total GPU seconds per environment

    for r in runs:
        try:
            df_r = _load_df(str(r.log_path),
                            str(r.epoch_csv_path) if r.epoch_csv_path else None)
            if not df_r.empty and "val_f1" in df_r.columns:
                curve_by_label[r.label] = df_r["val_f1"].dropna().round(4).tolist()
                run_best = _safe_max(df_r["val_f1"])
                if not pd.isna(run_best) and run_best > best_f1_global:
                    best_f1_global, best_run_label, best_run_df = run_best, r.label, df_r
            if not df_r.empty and "epoch_time" in df_r.columns and df_r["epoch_time"].notna().any():
                secs = float(df_r["epoch_time"].dropna().sum())
                total_gpu_h += secs / 3600
                gpu_secs_by_env[r.env] = gpu_secs_by_env.get(r.env, 0.0) + secs
                fastest_min = min(fastest_min, float(df_r["epoch_time"].dropna().mean()) / 60)
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

    # ── Row 1: varied charts — a time bar, the dataset split (pie) and the class
    #          imbalance treemap. Three different chart types, colourful summary ──
    meta = next((p for p in _META_CANDIDATES if Path(p).exists()), None)
    root = next((p for p in _ROOT_CANDIDATES if Path(p).exists()), None)
    dist = _dataset_dist(meta)

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.caption("GPU time by environment — hours")
            _time_by_env_bars(gpu_secs_by_env)
    with c2:
        with st.container(border=True):
            st.caption("Dataset split — train / val / test")
            _split_pie()
    with c3:
        with st.container(border=True):
            st.caption("Class imbalance — tile area = patches")
            _class_treemap(dist)

    # ── Row 2: active-run card (with mini F1/loss curves) + sample patches ───────
    a_left, a_right = st.columns([1.4, 1])
    with a_left:
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
    with a_right:
        with st.container(border=True):
            st.caption("Sample Sentinel-2 patches — RGB proxy")
            _photo_strip(meta, root)

    # ── All runs — selectable table (click a row → active run) ──────────────────
    st.markdown("#### All runs")
    st.caption("Click a row to make that run active. Dataset & import: **Dataset** section.")
    _all_runs_table(runs, curve_by_label)


def _dataset_dist(meta):
    """Per-class train counts — from the parquet if mounted, else an approximation."""
    dist = _load_class_distribution(str(meta)) if meta else None
    if dist is None:
        dist = class_distribution_approximate()
    return dist


def _group_of(class_name: str):
    """(group name, colour) for a class name, via its index in CLASS_NAMES."""
    idx = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else -1
    for gname, (idxs, color) in _CLASS_GROUPS.items():
        if idx in idxs:
            return gname, color
    return "Other", COLORS[0]


def _short_cls(name: str, n: int = 13) -> str:
    return name if len(name) <= n else name[:n - 1] + "…"


# Stable colour per environment for the GPU-time bar.
_ENV_COLOR = {"local": COLORS[0], "verode": COLORS[1], "kaggle": COLORS[2]}


def _env_color(env: str) -> str:
    return _ENV_COLOR.get(env, COLORS[3])


def _time_by_env_bars(gpu_secs_by_env: dict[str, float]) -> None:
    """Vertical bars of total GPU hours per environment — where the project's
    compute actually ran (local RTX 3060 Ti, Verode V100, Kaggle T4)."""
    if not gpu_secs_by_env:
        st.info("No timing data.")
        return
    items = sorted(gpu_secs_by_env.items(), key=lambda x: -x[1])
    envs = [e for e, _ in items]
    hrs = [s / 3600 for _, s in items]
    fig = go.Figure(go.Bar(
        x=[e.capitalize() for e in envs], y=hrs,
        marker=dict(color=[_env_color(e) for e in envs], line=dict(width=0)),
        text=[f"{h:.0f} h" for h in hrs], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.1f} GPU hours<extra></extra>",
    ))
    fig.update_layout(**_base_layout(145), showlegend=False, yaxis_title="GPU hours",
                      margin=dict(l=10, r=10, t=20, b=10))
    _show(fig, "ov_time_env_bars")


def _split_pie() -> None:
    """Pie of the dataset partition — train / val / test patch counts."""
    labels = ["Train", "Val", "Test"]
    vals = [SPLIT_SIZES["train"], SPLIT_SIZES["val"], SPLIT_SIZES["test"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, sort=False,
        marker=dict(colors=[COLORS[0], COLORS[1], COLORS[2]],
                    line=dict(width=1.5, color="white")),
        textinfo="label+percent", textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>%{value:,} patches · %{percent}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(145), showlegend=False,
                      margin=dict(l=6, r=6, t=10, b=6))
    _show(fig, "ov_split_pie")


def _class_treemap(dist) -> None:
    """Treemap of the 19 classes (tile area = #train patches) — the imbalance that
    caps macro-F1, in a third, distinct chart type."""
    d = dist.sort_values("train_count", ascending=False)
    colors = [_group_of(c)[1] for c in d["class"]]
    fig = go.Figure(go.Treemap(
        labels=[_short_cls(c, 16) for c in d["class"]], parents=[""] * len(d),
        values=d["train_count"],
        marker=dict(colors=colors, line=dict(width=1, color="white")),
        texttemplate="%{label}", textfont=dict(size=9), customdata=list(d["class"]),
        hovertemplate="<b>%{customdata}</b><br>%{value:,} patches · %{percentRoot}<extra></extra>",
        tiling=dict(pad=2),
    ))
    fig.update_layout(**_base_layout(145), margin=dict(t=6, l=0, r=0, b=0))
    _show(fig, "ov_class_treemap")


def _photo_strip(meta, root) -> None:
    """Real Sentinel-2 patches (RGB proxy), one per class — a small carousel paged
    5 at a time with ◀/▶ arrows (5·5·5·4 = all 19). Skipped when not mounted."""
    if not (meta and root):
        st.info("Dataset not mounted here — sample patches show on a machine with "
                "BigEarthNet-S2 available.")
        return
    try:
        _avg, gallery = _class_gallery(str(meta), str(root))
    except Exception:
        gallery = []
    if not gallery:
        st.info("No sample patches available.")
        return

    per_page = 5
    n = len(gallery)
    pages = (n + per_page - 1) // per_page
    page = st.session_state.get("ov_photo_page", 0) % pages

    nav1, nav2, nav3 = st.columns([1, 5, 1])
    if nav1.button("◀", key="ov_photo_prev", help="Previous classes", width="stretch"):
        st.session_state["ov_photo_page"] = (page - 1) % pages
        st.rerun()
    lo, hi = page * per_page, min(page * per_page + per_page, n)
    nav2.markdown(
        f"<div style='text-align:center;font-size:0.72rem;color:#6B7280;"
        f"padding-top:0.4rem'>Classes {lo + 1}–{hi} of {n}</div>",
        unsafe_allow_html=True)
    if nav3.button("▶", key="ov_photo_next", help="Next classes", width="stretch"):
        st.session_state["ov_photo_page"] = (page + 1) % pages
        st.rerun()

    cols = st.columns(per_page, gap="small")   # fixed 5 columns → stable layout
    for col, (cls, cnt, pct, img, labels) in zip(cols, gallery[lo:hi]):
        col.image(img, use_container_width=True)
        title = " · ".join(labels).replace("'", "")
        col.markdown(
            f"<div title='{title}' style='font-size:0.6rem;line-height:1.05;"
            f"height:2.2rem;overflow:hidden'>{_short_cls(cls, 14)}</div>",
            unsafe_allow_html=True)


def _kpi_strip(items: list[tuple[str, str]]) -> None:
    """One dense row of stat cards (much more compact than st.metric cards)."""
    cells = "".join(
        f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>"
        for l, v in items
    )
    st.markdown(f"<div class='kpi-strip'>{cells}</div>", unsafe_allow_html=True)


def _run_highlight(df: pd.DataFrame, anomalies_path=None, compact: bool = False) -> None:
    """Metric strip + a one-line verdict (card body). With ``compact=False`` it
    also draws the two mini F1/loss curves; the Overview uses ``compact=True`` to
    fit one screen — the full curves live in Run results."""
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

    if not compact:
        cc1, cc2 = st.columns(2)
        with cc1:
            fig = go.Figure()
            if "train_f1" in df.columns:
                fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_f1"], name="Train",
                                         line=dict(color=COLORS[0], width=2)))
            fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_f1"], name="Val",
                                     line=dict(color=COLORS[1], width=2)))
            fig.update_layout(**_base_layout(108, "F1 (macro)"), xaxis_title="Epoch", yaxis_title="F1")
            _show(fig, "hub_f1")
        with cc2:
            if "val_loss" in df.columns:
                fig = go.Figure()
                if "train_loss" in df.columns:
                    fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], name="Train",
                                             line=dict(color=COLORS[0], width=2)))
                fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], name="Val",
                                         line=dict(color=COLORS[3], width=2)))
                fig.update_layout(**_base_layout(108, "Loss (BCE)"), xaxis_title="Epoch", yaxis_title="Loss")
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
    # Bounded height: ~8 rows are visible, the rest scroll *inside* the table so
    # the page itself stays within one screen (no page scroll on the Overview).
    _table_h = min(34 + 30 * len(ov_df), 160)
    event = st.dataframe(
        ov_df,
        use_container_width=True, hide_index=True, height=_table_h,
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

