"""Streamlit web dashboard — Training Dashboard v5."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image

from src.web.batch_parser import parse_batch_csv
from src.web.confusion_matrix_parser import get_matrix_for_epoch, parse_confusion_matrix_csv
from src.web.dataset_stats import (
    CLASS_NAMES, SPLIT_SIZES,
    class_distribution_approximate, class_distribution_from_parquet,
    cooccurrence_from_perclass, get_country_distribution,
)
from src.web.feasibility_comparison import build_comparison
from src.web.feasibility_parser import parse_feasibility_csv
from src.web.log_parser import parse_log
from src.web.model_explorer import (
    ALL_FAMILIES, CURATED_MODELS, compare_models, get_model_stats,
)
from src.web.perclass_parser import parse_perclass_csv
from src.web.run_registry import RunInfo, discover_feasibility_csvs, discover_runs
from src.web.system_monitor import get_snapshot

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#64748b", "#ec4899", "#94a3b8"]

st.set_page_config(
    page_title="Training Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { min-width: 240px; max-width: 260px; }
  .block-container { padding-top: 4rem; padding-left: 1.5rem; padding-right: 1.5rem; }
  h1 { font-size: 1.4rem; font-weight: 600; }
  h2 { font-size: 1.1rem; font-weight: 600; margin-top: 1.2rem; }
  h3 { font-size: 0.95rem; font-weight: 600; }
  [data-testid="stMetricValue"] { font-size: 1.1rem; }

  /* Tab bar: allow horizontal scroll on narrow screens, shrink padding */
  [data-baseweb="tab-list"] {
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    scrollbar-width: thin;
    gap: 0 !important;
  }
  [data-baseweb="tab-list"]::-webkit-scrollbar { height: 3px; }
  [data-baseweb="tab-list"]::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

  [data-baseweb="tab"] {
    white-space: nowrap !important;
    font-size: 0.82rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
    min-width: unset !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_df(log_path: str, epoch_csv: str | None) -> pd.DataFrame:
    if epoch_csv and Path(epoch_csv).exists():
        df = pd.read_csv(epoch_csv)
        if not df.empty:
            if "epoch_time_s" in df.columns:
                df = df.rename(columns={"epoch_time_s": "epoch_time"})
            return df
    return parse_log(Path(log_path))


@st.cache_data(ttl=30)
def _load_batch(csv_path: str) -> pd.DataFrame:
    return parse_batch_csv(Path(csv_path))


@st.cache_data(ttl=30)
def _load_perclass(csv_path: str) -> pd.DataFrame:
    return parse_perclass_csv(Path(csv_path))


@st.cache_data(ttl=60)
def _get_runs() -> list[RunInfo]:
    return discover_runs(ROOT)


@st.cache_data(ttl=60)
def _get_feasibility_csvs() -> list[Path]:
    return discover_feasibility_csvs(ROOT)


def _safe_max(series: "pd.Series") -> float:
    """Return max of series, NaN if all values are NA."""
    valid = series.dropna()
    return float(valid.max()) if not valid.empty else float("nan")


def _safe_idxmax(series: "pd.Series"):
    """Return idxmax of series, None if all values are NA."""
    valid = series.dropna()
    return valid.idxmax() if not valid.empty else None


def _safe_val_at_best(df: "pd.DataFrame", metric_col: str, target_col: str):
    """Return value of target_col at the row where metric_col is maximum."""
    if metric_col not in df.columns or target_col not in df.columns:
        return None
    idx = _safe_idxmax(df[metric_col])
    if idx is None:
        return None
    v = df.loc[idx, target_col]
    return None if pd.isna(v) else v


# ── Helpers ────────────────────────────────────────────────────────────────────


def _throughput_col(df: pd.DataFrame) -> str | None:
    """Return the throughput column name, handling legacy and new CSV formats."""
    for col in ("imgs_per_s_train", "imgs_per_s"):
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _metric_fig(
    df: pd.DataFrame,
    col_train: str,
    col_val: str,
    title: str,
    y_label: str,
    color_train: str = COLORS[0],
    color_val: str = COLORS[1],
    extra_traces: list | None = None,
    height: int = 320,
) -> go.Figure:
    fig = go.Figure()
    if col_train in df.columns and df[col_train].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_train], name="Train",
            mode="lines+markers", line=dict(color=color_train, width=2),
            marker=dict(size=4),
        ))
    if col_val in df.columns and df[col_val].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_val], name="Val",
            mode="lines+markers", line=dict(color=color_val, width=2),
            marker=dict(size=4),
        ))
    for tr in (extra_traces or []):
        fig.add_trace(tr)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Epoch", yaxis_title=y_label,
        height=height, margin=dict(l=50, r=16, t=36, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
    )
    return fig


def _overlay_fig(
    dfs: list[tuple[str, pd.DataFrame]],
    col: str,
    title: str,
    y_label: str,
    height: int = 340,
) -> go.Figure:
    fig = go.Figure()
    for i, (label, df) in enumerate(dfs):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df[col],
                name=label[:30], mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                marker=dict(size=4),
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Epoch", yaxis_title=y_label,
        height=height, margin=dict(l=50, r=16, t=36, b=40),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
    )
    return fig


def _base_layout(height: int = 320, title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13)),
        height=height, margin=dict(l=50, r=16, t=36, b=40),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
    )


def _gpu_usage() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) < 5:
            return None
        return {
            "name": parts[0], "mem_used_mb": int(parts[1]),
            "mem_total_mb": int(parts[2]), "util_pct": int(parts[3]),
            "temp_c": int(parts[4]),
        }
    except Exception:
        return None


def _parse_log_progress(log_path: Path) -> dict:
    result = {"epoch": 0, "epochs": 0, "last_val_f1": None, "last_val_loss": None}
    try:
        import re
        lines = log_path.read_text(errors="replace").splitlines()
        for line in reversed(lines):
            if "Epoch" in line and "/" in line:
                m = re.search(r"Epoch\s+(\d+)/(\d+)", line)
                if m:
                    result["epoch"] = int(m.group(1))
                    result["epochs"] = int(m.group(2))
                    break
        for line in reversed(lines):
            if "val_f1" in line or "val=0." in line:
                m = re.search(r"val_f1[=\s]+([\d.]+)", line)
                if m:
                    result["last_val_f1"] = float(m.group(1))
                m2 = re.search(r"val_loss[=\s]+([\d.]+)", line)
                if m2:
                    result["last_val_loss"] = float(m2.group(1))
                break
    except Exception:
        pass
    return result


def _read_log_tail(log_path: Path, n: int = 40) -> str:
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def _detect_anomalies(log_path: Path) -> list[str]:
    keywords = ["EXPLODE", "VANISH", "DEAD", "OOM", "explosivo", "evanescente", "muertas"]
    hits: list[str] = []
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            if any(kw in line for kw in keywords):
                hits.append(line.strip())
    except Exception:
        pass
    return hits


def _get_configs() -> list[str]:
    cfg_dir = ROOT / "configs"
    if not cfg_dir.exists():
        return []
    return sorted(p.name for p in cfg_dir.glob("*.yaml"))


def _launch_process(cmd: str, placeholder) -> int:
    output_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(ROOT),
        )
        for raw in proc.stdout:  # type: ignore[union-attr]
            output_lines.append(raw.rstrip())
            placeholder.code("\n".join(output_lines[-120:]), language="text")
        proc.wait()
        return proc.returncode
    except Exception as exc:
        placeholder.error(str(exc))
        return -1


def _color_f1_cell(v: float) -> str:
    if v >= 0.6:
        return "background-color: #d1fae5; color: #065f46"
    if v >= 0.3:
        return "background-color: #fef3c7; color: #92400e"
    return "background-color: #fee2e2; color: #991b1b"


# ── Sidebar ────────────────────────────────────────────────────────────────────

runs = _get_runs()

with st.sidebar:
    st.markdown("### Training Dashboard")
    st.markdown("---")

    if not runs:
        st.warning("No runs found in logs/.")
        selected_run = None
    else:
        trace_filter = st.selectbox("Trace mode", ["all", "simple", "deep"])
        filtered = [r for r in runs if trace_filter == "all" or r.trace_mode == trace_filter]

        if not filtered:
            st.warning("No runs match the filter.")
            selected_run = None
        else:
            run_labels = {r.label: r for r in filtered}
            selected_label = st.selectbox("Run", list(run_labels.keys()))
            run = run_labels[selected_label]
            selected_run = run

            st.markdown("---")
            has_csv = run.epoch_csv_path is not None and run.epoch_csv_path.exists()
            model_label = run.model or "—"
            st.caption(
                f"**Log:** {run.log_path.name}  \n"
                f"**Env:** {run.env}  \n"
                f"**Mode:** {run.mode}  \n"
                f"**Model:** {model_label}  \n"
                f"**Trace:** {run.trace_mode}  \n"
                f"**Epoch CSV:** {'yes' if has_csv else 'no'}  \n"
                f"**Batch CSV:** {'yes' if run.batch_csv_path else 'no'}  \n"
                f"**Per-class CSV:** {'yes' if run.perclass_csv_path else 'no'}"
            )

    st.markdown("---")
    st.markdown("**Live Monitor**")
    live_mode = st.toggle("Auto-refresh", key="live_mode")
    refresh_interval = st.slider("Interval (s)", 5, 60, 10, disabled=not live_mode)

# ── Tabs ───────────────────────────────────────────────────────────────────────

(
    tab_overview, tab_system, tab_dataset, tab_models,
    tab_curves, tab_perclass, tab_batch, tab_compare, tab_ddp,
    tab_feasibility, tab_time, tab_info,
    tab_launcher, tab_live,
) = st.tabs([
    "Overview", "System", "Dataset", "Models",
    "Curves", "Per-class", "Batch", "Compare", "DDP Analysis",
    "Feasibility", "Time", "Info", "Launcher", "Live",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 0 — Overview
# ═══════════════════════════════════════════════════════════════════════════════

# BigEarthNet-S2 class groups for confusion matrix coloring
_CLASS_GROUPS = {
    "Urban":       ([0, 1],         "#6b7280"),
    "Agricultural":([ 2, 3, 4, 5, 6, 7], "#d97706"),
    "Forest":      ([8, 9, 10, 13], "#16a34a"),
    "Scrub/grass": ([11, 12],       "#84cc16"),
    "Bare/coastal":([14],           "#92400e"),
    "Wetlands":    ([15, 16],       "#0891b2"),
    "Water":       ([17, 18],       "#1d4ed8"),
}
_CLASS_GROUP_OF: dict[int, str] = {
    idx: name for name, (idxs, _) in _CLASS_GROUPS.items() for idx in idxs
}
_CLASS_GROUP_COLOR: dict[int, str] = {
    idx: color for name, (idxs, color) in _CLASS_GROUPS.items() for idx in idxs
}


with tab_overview:
    st.markdown("## Project overview")

    # ── Global stats ──────────────────────────────────────────────────────────
    total_runs = len(runs)
    best_f1_global = float("-inf")
    best_run_label = "—"
    total_gpu_h = 0.0

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
                s = df_r["epoch_time"].dropna().sum()
                total_gpu_h += float(s) / 3600
        except Exception:
            pass

    feasibility_csvs = _get_feasibility_csvs()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total runs", total_runs)
    c2.metric("Best Val F1", f"{best_f1_global:.4f}" if best_f1_global > float('-inf') else "—")
    c3.metric("Best run", best_run_label[:30] if best_run_label != "—" else "—")
    c4.metric("Total GPU time", f"{total_gpu_h:.1f} h")
    c5.metric("Feasibility reports", len(feasibility_csvs))

    st.markdown("---")

    # ── Recent runs table ─────────────────────────────────────────────────────
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
            best_ep = int(best_ep_v) if best_ep_v is not None else "—"
            n_ep = len(df_r)
            dur_s = df_r["epoch_time"].dropna().sum() if "epoch_time" in df_r.columns else float("nan")
            dur_str = f"{int(dur_s//3600)}h {int((dur_s%3600)//60)}m" if not pd.isna(dur_s) else "—"
            energy_wh = df_r["energy_eval_wh"].sum() if "energy_eval_wh" in df_r.columns and df_r["energy_eval_wh"].notna().any() else None
            overview_rows.append({
                "Run": r.label[:55],
                "Env": r.env,
                "Model": r.model or "—",
                "Trace": r.trace_mode,
                "Epochs": n_ep,
                "Best Val F1": round(run_best_f1, 4),
                "Best epoch": best_ep,
                "Duration": dur_str,
                "Energy eval (Wh)": f"{energy_wh:.0f}" if energy_wh else "—",
            })
        except Exception:
            pass

    if overview_rows:
        ov_df = pd.DataFrame(overview_rows)
        st.dataframe(
            ov_df.style.background_gradient(subset=["Best Val F1"], cmap="RdYlGn", vmin=0.4, vmax=0.75),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No runs with parseable metrics found.")

    # ── Mini training curve of selected run ───────────────────────────────────
    if selected_run is not None:
        st.markdown("---")
        st.markdown(f"### Selected run: {selected_run.label}")
        try:
            df_sel = _load_df(
                str(selected_run.log_path),
                str(selected_run.epoch_csv_path) if selected_run.epoch_csv_path else None,
            )
            if not df_sel.empty and "val_f1" in df_sel.columns:
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_mini = go.Figure()
                    fig_mini.add_trace(go.Scatter(
                        x=df_sel["epoch"], y=df_sel["train_f1"],
                        name="Train F1", line=dict(color=COLORS[0], width=2),
                    ))
                    fig_mini.add_trace(go.Scatter(
                        x=df_sel["epoch"], y=df_sel["val_f1"],
                        name="Val F1", line=dict(color=COLORS[1], width=2),
                    ))
                    fig_mini.update_layout(**_base_layout(220, "F1 curve"),
                                          xaxis_title="Epoch", yaxis_title="F1")
                    st.plotly_chart(fig_mini, use_container_width=True)
                with col_b:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        x=df_sel["epoch"], y=df_sel["train_loss"],
                        name="Train loss", line=dict(color=COLORS[0], width=2),
                    ))
                    fig_loss.add_trace(go.Scatter(
                        x=df_sel["epoch"], y=df_sel["val_loss"],
                        name="Val loss", line=dict(color=COLORS[3], width=2),
                    ))
                    fig_loss.update_layout(**_base_layout(220, "Loss curve"),
                                           xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig_loss, use_container_width=True)
        except Exception:
            st.info("Could not render mini chart for this run.")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab — System Monitor
# ═══════════════════════════════════════════════════════════════════════════════

with tab_system:
    st.markdown("## System monitor")
    auto_ref = st.sidebar.toggle("System auto-refresh", key="sys_refresh", value=False)
    ref_int = st.sidebar.slider("Refresh (s)", 2, 30, 5, key="sys_ref_int",
                                disabled=not auto_ref)

    snap = get_snapshot(disk_paths=["/", "/home", "/media"])

    # ── CPU ──────────────────────────────────────────────────────────────────
    st.markdown("### CPU")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Usage", f"{snap.cpu.usage_pct:.1f}%")
    sc2.metric("Logical cores", snap.cpu.count_logical)
    sc3.metric("Physical cores", snap.cpu.count_physical)
    sc4.metric("Frequency", f"{snap.cpu.freq_mhz:.0f} MHz" if snap.cpu.freq_mhz else "—")
    st.progress(snap.cpu.usage_pct / 100)

    # ── RAM ──────────────────────────────────────────────────────────────────
    st.markdown("### RAM")
    sr1, sr2, sr3, sr4 = st.columns(4)
    sr1.metric("Used", f"{snap.ram.used_gb:.1f} GB")
    sr2.metric("Total", f"{snap.ram.total_gb:.1f} GB")
    sr3.metric("Available", f"{snap.ram.available_gb:.1f} GB")
    sr4.metric("Usage %", f"{snap.ram.percent:.1f}%")
    st.progress(snap.ram.percent / 100)
    if snap.ram.swap_total_gb > 0:
        st.caption(
            f"Swap: {snap.ram.swap_used_gb:.1f} / {snap.ram.swap_total_gb:.1f} GB"
        )

    # ── GPU ──────────────────────────────────────────────────────────────────
    st.markdown("### GPU")
    if snap.gpus:
        for gpu in snap.gpus:
            mem_pct = gpu.mem_used_mb / gpu.mem_total_mb * 100 if gpu.mem_total_mb else 0
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric(f"GPU {gpu.index}", gpu.name[:28])
            g2.metric("VRAM used", f"{gpu.mem_used_mb / 1024:.1f} GB")
            g3.metric("VRAM total", f"{gpu.mem_total_mb / 1024:.1f} GB")
            g4.metric("Utilization", f"{gpu.util_pct}%")
            g5.metric("Temp", f"{gpu.temp_c}°C")
            st.progress(mem_pct / 100,
                        text=f"VRAM {gpu.mem_used_mb}/{gpu.mem_total_mb} MB ({mem_pct:.1f}%)")
            if gpu.power_w is not None:
                limit_str = f" / {gpu.power_limit_w:.0f} W" if gpu.power_limit_w else ""
                st.caption(f"Power draw: {gpu.power_w:.1f} W{limit_str}")
    else:
        st.info("No GPU detected (nvidia-smi not available).")

    # ── Disk ─────────────────────────────────────────────────────────────────
    st.markdown("### Disk")
    if snap.disks:
        disk_cols = st.columns(len(snap.disks))
        for col, disk in zip(disk_cols, snap.disks):
            col.metric(disk.path, f"{disk.free_gb:.1f} GB free")
            col.progress(disk.percent / 100,
                         text=f"{disk.used_gb:.1f} / {disk.total_gb:.1f} GB ({disk.percent:.1f}%)")
    else:
        st.info("Could not read disk usage.")

    # ── Network ──────────────────────────────────────────────────────────────
    st.markdown("### Network (cumulative since boot)")
    nn1, nn2 = st.columns(2)
    nn1.metric("Sent", f"{snap.network.bytes_sent_mb / 1024:.2f} GB")
    nn2.metric("Received", f"{snap.network.bytes_recv_mb / 1024:.2f} GB")

    if auto_ref:
        time.sleep(ref_int)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# Tab — Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dataset:
    st.markdown("## Dataset Explorer — BigEarthNet-S2 v2.0")

    # Try to load metadata from config paths
    meta_path: Path | None = None
    for candidate in [
        "/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet",
        "/home/bejeque/alu0101317038/datasets/bigearthnet/metadata.parquet",
    ]:
        if Path(candidate).exists():
            meta_path = Path(candidate)
            break

    # ── Split sizes ───────────────────────────────────────────────────────────
    st.markdown("### Dataset splits")
    ds1, ds2, ds3, ds4 = st.columns(4)
    ds1.metric("Train", f"{SPLIT_SIZES['train']:,}")
    ds2.metric("Validation", f"{SPLIT_SIZES['val']:,}")
    ds3.metric("Test", f"{SPLIT_SIZES['test']:,}")
    total = sum(SPLIT_SIZES.values())
    ds4.metric("Total patches", f"{total:,}")

    fig_splits = go.Figure(go.Pie(
        labels=list(SPLIT_SIZES.keys()),
        values=list(SPLIT_SIZES.values()),
        hole=0.4,
        marker_colors=[COLORS[0], COLORS[2], COLORS[1]],
    ))
    fig_splits.update_layout(**_base_layout(220, "Split distribution"), showlegend=True)
    st.plotly_chart(fig_splits, use_container_width=True)

    # ── Class distribution ────────────────────────────────────────────────────
    st.markdown("### Class distribution (training split)")
    if meta_path:
        dist_df = class_distribution_from_parquet(meta_path)
        st.caption(f"Source: {meta_path}")
    else:
        dist_df = None
        st.caption("metadata.parquet not found — using approximate statistics.")

    if dist_df is None:
        dist_df = class_distribution_approximate()

    dist_df = dist_df.sort_values("train_count", ascending=True)
    dist_df["color"] = dist_df["train_count"].apply(
        lambda v: COLORS[3] if v < 5000 else (COLORS[1] if v < 15000 else COLORS[2])
    )

    fig_dist = go.Figure(go.Bar(
        y=dist_df["class"], x=dist_df["train_count"],
        orientation="h",
        marker_color=dist_df["color"],
        text=dist_df["train_count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
    ))
    fig_dist.update_layout(
        **_base_layout(560, "Samples per class (train)"),
        xaxis_title="Sample count", yaxis_title="",
        margin=dict(l=230, r=80, t=40, b=40),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption(
        "Red = rare (<5K samples), Orange = moderate (<15K), Green = frequent. "
        "Rare classes dominate the F1 macro ceiling."
    )

    # ── Class imbalance analysis ──────────────────────────────────────────────
    st.markdown("### Class imbalance")
    max_c = dist_df["train_count"].max()
    min_c = dist_df["train_count"].min()
    ratio = max_c / min_c if min_c > 0 else float("inf")
    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Most frequent class", dist_df.iloc[-1]["class"][:30])
    ci2.metric("Rarest class", dist_df.iloc[0]["class"][:30])
    ci3.metric("Imbalance ratio", f"{ratio:.1f}×")

    # ── Country distribution ──────────────────────────────────────────────────
    if meta_path:
        country_counts = get_country_distribution(meta_path)
        if country_counts is not None and not country_counts.empty:
            st.markdown("### Country distribution (train)")
            top_n = country_counts.head(15)
            fig_c = px.bar(
                x=top_n.values, y=top_n.index, orientation="h",
                labels={"x": "Patches", "y": "Country"},
                color=top_n.values,
                color_continuous_scale="Blues",
            )
            fig_c.update_layout(**_base_layout(380, "Top countries by patch count"))
            st.plotly_chart(fig_c, use_container_width=True)

    # ── Per-class performance scatter ─────────────────────────────────────────
    st.markdown("### Class difficulty vs frequency")
    st.caption("Uses best available per-class CSV from any run.")

    perclass_csvs_all = list(ROOT.rglob("perclass_metrics_*.csv"))
    if perclass_csvs_all:
        latest_pc = max(perclass_csvs_all, key=lambda p: p.stat().st_mtime)
        pc_df = parse_perclass_csv(latest_pc)
        if not pc_df.empty:
            last_ep = pc_df["epoch"].max()
            ep_pc = pc_df[pc_df["epoch"] == last_ep].copy()
            ep_pc = ep_pc.merge(dist_df[["class", "train_count"]],
                                left_on="class_name", right_on="class", how="left")
            fig_sc = px.scatter(
                ep_pc, x="train_count", y="f1",
                text="class_name", color="f1",
                color_continuous_scale="RdYlGn",
                range_color=[0, 1],
                labels={"train_count": "Training samples", "f1": "Val F1"},
                title=f"F1 vs class frequency (epoch {last_ep})",
            )
            fig_sc.update_traces(textposition="top center", textfont_size=8)
            fig_sc.update_layout(**_base_layout(440), showlegend=False)
            st.plotly_chart(fig_sc, use_container_width=True)
            st.caption(
                "Classes with few training samples tend to have low F1. "
                "Points near the bottom-left are the hardest to improve."
            )
    else:
        st.info("No per-class CSV found. Run a training with `--layers confusion`.")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab — Model Explorer
# ═══════════════════════════════════════════════════════════════════════════════

with tab_models:
    st.markdown("## Model Explorer")
    st.caption("Browse timm models, compare parameters and VRAM requirements.")

    # ── Family selector ───────────────────────────────────────────────────────
    col_fam, col_bs = st.columns([3, 1])
    with col_fam:
        selected_families = st.multiselect(
            "Model families", ALL_FAMILIES, default=["ViT", "ResNet", "EfficientNet"],
        )
    with col_bs:
        cmp_batch = st.selectbox("Batch size for VRAM estimate", [4, 8, 16, 32, 64, 128], index=3)

    candidate_models = []
    for fam in selected_families:
        candidate_models.extend(CURATED_MODELS.get(fam, []))

    extra_model = st.text_input(
        "Add custom timm model (any valid ID)", placeholder="e.g. convnext_large"
    )
    if extra_model.strip():
        candidate_models.append(extra_model.strip())

    if not candidate_models:
        st.info("Select at least one family.")
    else:
        with st.spinner("Loading model stats (cached after first run)…"):
            rows = compare_models(candidate_models, [cmp_batch], num_classes=19)

        if not rows:
            st.warning("Could not load any model.")
        else:
            cmp_df = pd.DataFrame(rows)
            vram_col = f"VRAM est. bs={cmp_batch} (GB)"

            # Color VRAM column: green<4GB, orange 4-8, red>8
            def _color_vram(v):
                try:
                    fv = float(v)
                    if fv <= 4:
                        return "background-color: #dcfce7"
                    if fv <= 8:
                        return "background-color: #fef9c3"
                    return "background-color: #fee2e2"
                except (ValueError, TypeError):
                    return ""

            styled_cmp = cmp_df.style
            if vram_col in cmp_df.columns:
                styled_cmp = styled_cmp.map(_color_vram, subset=[vram_col])
            st.dataframe(styled_cmp, use_container_width=True, hide_index=True)
            st.caption("Green = fits in 4 GB | Orange = 4–8 GB | Red = >8 GB (RTX 3060 Ti limit)")

            # ── Bubble chart: params vs FLOPs, size=VRAM ─────────────────────
            st.markdown("### Parameters vs FLOPs")
            plot_df = cmp_df[cmp_df["FLOPs (MFLOPs)"] != "—"].copy()
            if not plot_df.empty:
                plot_df["FLOPs (MFLOPs)"] = pd.to_numeric(plot_df["FLOPs (MFLOPs)"], errors="coerce")
                plot_df[vram_col] = pd.to_numeric(plot_df.get(vram_col, 0), errors="coerce").fillna(1)
                fig_bubble = px.scatter(
                    plot_df,
                    x="FLOPs (MFLOPs)", y="Params (M)",
                    size=vram_col, color="Family",
                    text="Model", hover_name="Model",
                    size_max=40,
                    labels={"FLOPs (MFLOPs)": "FLOPs per image (MFLOPs)",
                            "Params (M)": "Parameters (M)"},
                )
                fig_bubble.update_traces(textposition="top center", textfont_size=8)
                fig_bubble.update_layout(**_base_layout(420, "Model complexity landscape"),
                                         showlegend=True)
                st.plotly_chart(fig_bubble, use_container_width=True)
                st.caption("Bubble size = estimated VRAM at selected batch size.")

            # ── VRAM bar chart by batch size ──────────────────────────────────
            st.markdown("### VRAM requirements across batch sizes")
            vram_models = candidate_models[:8]  # limit for readability
            with st.spinner("Computing VRAM estimates…"):
                vram_rows = compare_models(vram_models, [4, 8, 16, 32, 64, 128])

            if vram_rows:
                vram_df = pd.DataFrame(vram_rows)
                vram_cols = [c for c in vram_df.columns if c.startswith("VRAM")]
                fig_vram = go.Figure()
                for col in vram_cols:
                    bs_val = col.split("bs=")[1].split(" ")[0]
                    fig_vram.add_trace(go.Bar(
                        name=f"bs={bs_val}",
                        x=vram_df["Model"],
                        y=pd.to_numeric(vram_df[col], errors="coerce"),
                    ))
                fig_vram.add_hline(y=8, line_dash="dash", line_color="red",
                                   annotation_text="RTX 3060 Ti (8 GB)")
                fig_vram.add_hline(y=32, line_dash="dash", line_color="orange",
                                   annotation_text="V100 (32 GB)")
                fig_vram.update_layout(
                    **_base_layout(380, "Estimated VRAM (GB) by batch size"),
                    barmode="group",
                    xaxis_tickangle=30,
                    yaxis_title="GB",
                )
                st.plotly_chart(fig_vram, use_container_width=True)

            # ── Quick launch integration ──────────────────────────────────────
            st.markdown("### Quick launch")
            selected_for_launch = st.selectbox(
                "Select model to pre-fill Launcher", [r["Model"] for r in rows],
            )
            if selected_for_launch:
                st.info(
                    f"Go to the **Launcher** tab — the model `{selected_for_launch}` "
                    f"is remembered. Paste it in the model field."
                )
                st.session_state["preselected_model"] = selected_for_launch

# ═══════════════════════════════════════════════════════════════════════════════
# Tab — DDP Analysis
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ddp:
    st.markdown("## DDP Analysis — Single-GPU vs Distributed")
    st.caption(
        "Compares single-GPU and DDP runs of the same model to measure "
        "actual speedup, efficiency, and scaling behaviour."
    )

    all_runs_ddp = _get_runs()
    if not all_runs_ddp:
        st.info("No runs found.")
    else:
        # Detect single vs ddp by path mode
        single_runs = [r for r in all_runs_ddp if r.mode == "single"]
        ddp_runs = [r for r in all_runs_ddp if r.mode == "ddp"]

        da1, da2, da3 = st.columns(3)
        da1.metric("Single-GPU runs", len(single_runs))
        da2.metric("DDP runs", len(ddp_runs))
        da3.metric("Total runs", len(all_runs_ddp))

        if not ddp_runs:
            st.info(
                "No DDP runs yet. Launch a distributed training with "
                "`scripts/train_ddp.py` — results will appear here automatically."
            )
        else:
            st.markdown("### DDP runs")
            ddp_rows = []
            for r in ddp_runs:
                try:
                    ddf = _load_df(str(r.log_path),
                                   str(r.epoch_csv_path) if r.epoch_csv_path else None)
                    if ddf.empty:
                        continue
                    best_f1 = _safe_max(ddf["val_f1"]) if "val_f1" in ddf.columns else float("nan")
                    avg_epoch_s = ddf["epoch_time"].dropna().mean() if "epoch_time" in ddf.columns and ddf["epoch_time"].notna().any() else None
                    ddp_rows.append({
                        "Run": r.label[:50], "Model": r.model or "—",
                        "Env": r.env, "Best Val F1": round(best_f1, 4),
                        "Epochs": len(ddf),
                        "Avg epoch (min)": round(avg_epoch_s / 60, 1) if avg_epoch_s else "—",
                    })
                except Exception:
                    pass
            if ddp_rows:
                st.dataframe(pd.DataFrame(ddp_rows), use_container_width=True, hide_index=True)

        # ── Speedup comparison ────────────────────────────────────────────────
        if single_runs and ddp_runs:
            st.markdown("### Speedup analysis")

            col_s, col_d = st.columns(2)
            with col_s:
                single_lbl = st.selectbox(
                    "Single-GPU run",
                    [r.label for r in single_runs], key="ddp_single_sel",
                )
            with col_d:
                ddp_lbl = st.selectbox(
                    "DDP run", [r.label for r in ddp_runs], key="ddp_ddp_sel",
                )

            r_single = next(r for r in single_runs if r.label == single_lbl)
            r_ddp = next(r for r in ddp_runs if r.label == ddp_lbl)

            df_s = _load_df(str(r_single.log_path),
                            str(r_single.epoch_csv_path) if r_single.epoch_csv_path else None)
            df_d = _load_df(str(r_ddp.log_path),
                            str(r_ddp.epoch_csv_path) if r_ddp.epoch_csv_path else None)

            if not df_s.empty and not df_d.empty:
                avg_s = df_s["epoch_time"].mean() if "epoch_time" in df_s.columns and df_s["epoch_time"].notna().any() else None
                avg_d = df_d["epoch_time"].mean() if "epoch_time" in df_d.columns and df_d["epoch_time"].notna().any() else None

                su1, su2, su3, su4 = st.columns(4)
                su1.metric("Single-GPU epoch", f"{avg_s/60:.1f} min" if avg_s else "—")
                su2.metric("DDP epoch", f"{avg_d/60:.1f} min" if avg_d else "—")
                if avg_s and avg_d and avg_d > 0:
                    speedup = avg_s / avg_d
                    su3.metric("Actual speedup", f"{speedup:.2f}×")
                    world_size_ddp = 2  # assume 2 GPUs by default
                    efficiency = speedup / world_size_ddp * 100
                    su4.metric("Scaling efficiency", f"{efficiency:.1f}%")

                # F1 comparison overlay
                fig_ddp_f1 = go.Figure()
                if "val_f1" in df_s.columns:
                    fig_ddp_f1.add_trace(go.Scatter(
                        x=df_s["epoch"], y=df_s["val_f1"],
                        name="Single-GPU Val F1", line=dict(color=COLORS[0], width=2),
                    ))
                if "val_f1" in df_d.columns:
                    fig_ddp_f1.add_trace(go.Scatter(
                        x=df_d["epoch"], y=df_d["val_f1"],
                        name="DDP Val F1", line=dict(color=COLORS[2], width=2),
                    ))
                fig_ddp_f1.update_layout(
                    **_base_layout(300, "Val F1: Single-GPU vs DDP"),
                    xaxis_title="Epoch", yaxis_title="Val F1",
                )
                st.plotly_chart(fig_ddp_f1, use_container_width=True)

                # Epoch time comparison
                if avg_s and avg_d:
                    fig_time_ddp = go.Figure()
                    if "epoch_time" in df_s.columns:
                        fig_time_ddp.add_trace(go.Scatter(
                            x=df_s["epoch"], y=df_s["epoch_time"] / 60,
                            name="Single-GPU", line=dict(color=COLORS[0], width=2),
                        ))
                    if "epoch_time" in df_d.columns:
                        fig_time_ddp.add_trace(go.Scatter(
                            x=df_d["epoch"], y=df_d["epoch_time"] / 60,
                            name="DDP", line=dict(color=COLORS[2], width=2),
                        ))
                    fig_time_ddp.update_layout(
                        **_base_layout(260, "Epoch time: Single-GPU vs DDP"),
                        xaxis_title="Epoch", yaxis_title="Minutes",
                    )
                    st.plotly_chart(fig_time_ddp, use_container_width=True)

                # Theoretical scaling chart
                st.markdown("### Theoretical vs actual scaling")
                world_sizes = [1, 2, 4, 8]
                theoretical = [avg_s / ws for ws in world_sizes] if avg_s else []
                if theoretical:
                    fig_scale = go.Figure()
                    fig_scale.add_trace(go.Scatter(
                        x=world_sizes, y=[t / 60 for t in theoretical],
                        name="Theoretical (100% efficiency)",
                        line=dict(color=COLORS[4], width=2, dash="dash"),
                        mode="lines+markers",
                    ))
                    if avg_d:
                        fig_scale.add_trace(go.Scatter(
                            x=[2], y=[avg_d / 60],
                            name="Actual DDP (2 GPUs)",
                            mode="markers",
                            marker=dict(color=COLORS[2], size=14, symbol="star"),
                        ))
                    fig_scale.update_layout(
                        **_base_layout(300, "Epoch time vs number of GPUs"),
                        xaxis_title="Number of GPUs",
                        yaxis_title="Minutes per epoch",
                        xaxis=dict(tickvals=world_sizes),
                    )
                    st.plotly_chart(fig_scale, use_container_width=True)
                    st.caption(
                        "The gap between theoretical and actual reflects communication "
                        "overhead, NFS bottleneck, and load imbalance."
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Training Curves
# ═══════════════════════════════════════════════════════════════════════════════

with tab_curves:
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
        df = _load_df(
            str(run.log_path),
            str(run.epoch_csv_path) if run.epoch_csv_path else None,
        )

        if df.empty:
            st.error("Could not parse any epochs from the selected run.")
        else:
            n_epochs = len(df)
            best_f1 = _safe_max(df["val_f1"]) if "val_f1" in df.columns else float("nan")
            _best_ep_v = _safe_val_at_best(df, "val_f1", "epoch")
            best_epoch = int(_best_ep_v) if _best_ep_v is not None else "—"
            best_thresh_f1 = (
                _safe_max(df["f1_at_threshold"])
                if "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any()
                else None
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Epochs", n_epochs)
            c2.metric("Best Val F1", f"{best_f1:.4f}" if not pd.isna(best_f1) else "—")
            c3.metric("Best epoch", best_epoch)
            if best_thresh_f1 is not None:
                c4.metric("F1 @ optimal threshold", f"{best_thresh_f1:.4f}")
            if "epoch_time" in df.columns and df["epoch_time"].notna().any():
                total_s = df["epoch_time"].sum()
                c5.metric("Total duration", f"{int(total_s//3600)}h {int((total_s%3600)//60)}m")

            src = "epoch_metrics CSV" if (run.epoch_csv_path and run.epoch_csv_path.exists()) else "log file"
            st.caption(f"Source: {src}")

            extra_thresh: list = []
            if "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any():
                extra_thresh = [go.Scatter(
                    x=df["epoch"], y=df["f1_at_threshold"],
                    name="F1 @ optimal threshold", mode="lines",
                    line=dict(color=COLORS[2], width=2, dash="dot"),
                )]

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    _metric_fig(df, "train_f1", "val_f1", "F1 (macro)", "F1",
                                extra_traces=extra_thresh),
                    use_container_width=True,
                )
                st.plotly_chart(
                    _metric_fig(df, "train_loss", "val_loss", "Loss (BCE)", "Loss"),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    _metric_fig(df, "train_acc", "val_acc", "Accuracy", "Accuracy",
                                color_train=COLORS[4], color_val=COLORS[5]),
                    use_container_width=True,
                )
                st.plotly_chart(
                    _metric_fig(df, "val_prec", "val_rec", "Precision & Recall (val)",
                                "Score", color_train=COLORS[2], color_val=COLORS[3]),
                    use_container_width=True,
                )

            if "epoch_time" in df.columns and df["epoch_time"].notna().any():
                et = df[["epoch", "epoch_time"]].dropna()
                fig_et = go.Figure(go.Bar(
                    x=et["epoch"], y=et["epoch_time"] / 60,
                    marker_color=COLORS[0], opacity=0.8,
                ))
                fig_et.update_layout(**_base_layout(240, "Time per epoch (min)"),
                                     xaxis_title="Epoch", yaxis_title="Minutes")
                st.plotly_chart(fig_et, use_container_width=True)

            # ── Energy charts ─────────────────────────────────────────────
            has_energy = (
                "energy_eval_wh" in df.columns and df["energy_eval_wh"].notna().any()
            )
            if has_energy:
                st.markdown("#### Energy consumption")
                e1, e2 = st.columns(2)
                with e1:
                    rows_e = []
                    for _, row in df.iterrows():
                        if pd.notna(row.get("energy_eval_wh")):
                            rows_e.append({"epoch": row["epoch"],
                                           "Eval (Wh)": row["energy_eval_wh"]})
                        if pd.notna(row.get("energy_train_j")):
                            rows_e[-1]["Train (Wh)"] = row["energy_train_j"] / 3600
                    if rows_e:
                        df_e = pd.DataFrame(rows_e)
                        fig_e = go.Figure()
                        if "Train (Wh)" in df_e.columns:
                            fig_e.add_trace(go.Bar(
                                x=df_e["epoch"], y=df_e["Train (Wh)"],
                                name="Train", marker_color=COLORS[0], opacity=0.85,
                            ))
                        fig_e.add_trace(go.Bar(
                            x=df_e["epoch"], y=df_e["Eval (Wh)"],
                            name="Eval", marker_color=COLORS[1], opacity=0.85,
                        ))
                        fig_e.update_layout(
                            **_base_layout(260, "Energy per epoch (Wh)"),
                            barmode="group", xaxis_title="Epoch", yaxis_title="Wh",
                        )
                        st.plotly_chart(fig_e, use_container_width=True)

                with e2:
                    power_cols = []
                    if "power_eval_w" in df.columns and df["power_eval_w"].notna().any():
                        power_cols.append(("Eval power (W)", "power_eval_w", COLORS[1]))
                    if "power_train_w" in df.columns and df["power_train_w"].notna().any():
                        power_cols.append(("Train power (W)", "power_train_w", COLORS[0]))
                    if power_cols:
                        fig_p = go.Figure()
                        for name, col, color in power_cols:
                            fig_p.add_trace(go.Scatter(
                                x=df["epoch"], y=df[col],
                                name=name, mode="lines+markers",
                                line=dict(color=color, width=2),
                            ))
                        fig_p.update_layout(
                            **_base_layout(260, "Average GPU power per epoch (W)"),
                            xaxis_title="Epoch", yaxis_title="Watts",
                        )
                        st.plotly_chart(fig_p, use_container_width=True)

                total_eval_wh = df["energy_eval_wh"].sum() if "energy_eval_wh" in df.columns else 0
                total_train_wh = df["energy_train_j"].sum() / 3600 if "energy_train_j" in df.columns and df["energy_train_j"].notna().any() else 0
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("Total eval energy", f"{total_eval_wh:.1f} Wh")
                if total_train_wh > 0:
                    ec2.metric("Total train energy", f"{total_train_wh:.1f} Wh")
                    ec3.metric("Total energy", f"{total_eval_wh + total_train_wh:.1f} Wh")

            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                "Download epoch_metrics.csv", csv_bytes,
                file_name="epoch_metrics.csv", mime="text/csv",
            )

            with st.expander("Full epoch table"):
                st.dataframe(df.set_index("epoch"), use_container_width=True)

            if run.plot_path and run.plot_path.exists():
                with st.expander("Saved PNG"):
                    st.image(str(run.plot_path), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Per-class Metrics
# ═══════════════════════════════════════════════════════════════════════════════

with tab_perclass:
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
        subtab_bars, subtab_trend, subtab_cm = st.tabs(
            ["By class", "Trend", "Confusion matrix"]
        )

        with subtab_bars:
            if run.perclass_csv_path and run.perclass_csv_path.exists():
                pcdf = _load_perclass(str(run.perclass_csv_path))
                epochs_available = sorted(pcdf["epoch"].unique().tolist())

                selected_ep = st.selectbox(
                    "Epoch", epochs_available, format_func=lambda e: f"Epoch {e}",
                )
                ep_df = pcdf[pcdf["epoch"] == selected_ep].copy()
                ep_df = ep_df.sort_values("f1", ascending=False)

                # Ranking table with color coding — use .map() (pandas 2+)
                styled = (
                    ep_df[["class_name", "f1", "precision", "recall"]]
                    .style
                    .map(_color_f1_cell, subset=["f1"])
                    .format({"f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"})
                )
                st.dataframe(styled, use_container_width=True, height=280)

                colors_f1 = [
                    COLORS[2] if v >= 0.6 else (COLORS[1] if v >= 0.3 else COLORS[3])
                    for v in ep_df["f1"]
                ]
                fig_pc = go.Figure()
                fig_pc.add_trace(go.Bar(
                    y=ep_df["class_name"], x=ep_df["precision"],
                    name="Precision", orientation="h",
                    marker_color=COLORS[0], opacity=0.8,
                ))
                fig_pc.add_trace(go.Bar(
                    y=ep_df["class_name"], x=ep_df["recall"],
                    name="Recall", orientation="h",
                    marker_color=COLORS[1], opacity=0.8,
                ))
                fig_pc.add_trace(go.Bar(
                    y=ep_df["class_name"], x=ep_df["f1"],
                    name="F1", orientation="h", marker_color=colors_f1,
                ))
                fig_pc.update_layout(
                    barmode="group",
                    title=dict(text=f"Per-class metrics — Epoch {selected_ep}", font=dict(size=13)),
                    xaxis_title="Score", xaxis=dict(range=[0, 1]),
                    height=600, margin=dict(l=200, r=16, t=36, b=40),
                    paper_bgcolor="white", plot_bgcolor="#f8fafc",
                )
                st.plotly_chart(fig_pc, use_container_width=True)

            elif run.perclass_paths:
                epoch_opts = [p.stem.split("_epoch")[-1] for p in run.perclass_paths]
                idx = st.selectbox("Epoch", range(len(run.perclass_paths)),
                                   format_func=lambda i: f"Epoch {epoch_opts[i]}")
                if run.perclass_paths[idx].exists():
                    st.image(Image.open(run.perclass_paths[idx]), use_container_width=True)
            else:
                st.info("No per-class data. Use `--layers confusion` to generate it.")

        with subtab_trend:
            if run.perclass_csv_path and run.perclass_csv_path.exists():
                pcdf = _load_perclass(str(run.perclass_csv_path))
                classes = sorted(pcdf["class_name"].unique().tolist())

                col_sel, col_met = st.columns([3, 1])
                with col_sel:
                    selected_classes = st.multiselect(
                        "Classes (max 8)", classes, default=classes[:4], max_selections=8,
                    )
                with col_met:
                    metric_sel = st.radio("Metric", ["f1", "precision", "recall"])

                if selected_classes:
                    fig_trend = go.Figure()
                    for i, cls in enumerate(selected_classes):
                        cdf = pcdf[pcdf["class_name"] == cls].sort_values("epoch")
                        fig_trend.add_trace(go.Scatter(
                            x=cdf["epoch"], y=cdf[metric_sel],
                            name=cls[:30], mode="lines+markers",
                            line=dict(color=COLORS[i % len(COLORS)], width=2),
                            marker=dict(size=4),
                        ))
                    fig_trend.update_layout(
                        **_base_layout(400, f"{metric_sel.capitalize()} by class over epochs"),
                        xaxis_title="Epoch",
                    )
                    fig_trend.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No per-class CSV for this run.")

        with subtab_cm:
            if run.confusion_matrix_csv_path and run.confusion_matrix_csv_path.exists():
                cm_df = parse_confusion_matrix_csv(run.confusion_matrix_csv_path)
                epochs_cm = sorted(cm_df["epoch"].unique().tolist())

                col_cm1, col_cm2 = st.columns([3, 1])
                with col_cm1:
                    selected_cm_ep = st.selectbox(
                        "Epoch", epochs_cm, format_func=lambda e: f"Epoch {e}",
                        key="cm_epoch_sel",
                    )
                with col_cm2:
                    cm_mode = st.radio("Mode", ["Normalized", "Absolute"], key="cm_mode")

                pivot = get_matrix_for_epoch(cm_df, selected_cm_ep)
                class_order = list(pivot.index)
                z_norm = pivot.reindex(index=class_order, columns=class_order).values
                n_classes = len(class_order)

                if cm_mode == "Absolute":
                    row_sums = z_norm.sum(axis=1, keepdims=True)
                    z_abs = (z_norm * row_sums).round().astype(int)
                    z_plot = z_abs.tolist()
                    text = [[str(v) if v > 0 else "" for v in row] for row in z_abs]
                    zmin, zmax, cb_title = 0, None, "Samples"
                else:
                    z_plot = z_norm.tolist()
                    text = [[f"{v:.2f}" if v >= 0.05 else "" for v in row] for row in z_norm]
                    zmin, zmax, cb_title = 0, 1, "P(pred j | true i)"

                # Build colored group shapes for the diagonal blocks
                # Map class names → group color using class index
                def _group_color_for_name(name: str) -> str:
                    for group_name, (idxs, color) in _CLASS_GROUPS.items():
                        for idx in idxs:
                            if str(idx) in name or any(
                                kw.lower() in name.lower()
                                for kw in group_name.split("/")
                            ):
                                return color
                    # fallback: match by position in class_order
                    pos = class_order.index(name) if name in class_order else -1
                    return _CLASS_GROUP_COLOR.get(pos, "#94a3b8")

                shapes = []
                # Draw colored rectangles on the diagonal per group
                for group_name, (idxs, color) in _CLASS_GROUPS.items():
                    positions = [
                        i for i, cls in enumerate(class_order)
                        if i in idxs
                    ]
                    if not positions:
                        continue
                    lo, hi = min(positions), max(positions)
                    shapes.append(dict(
                        type="rect",
                        x0=lo - 0.5, x1=hi + 0.5,
                        y0=lo - 0.5, y1=hi + 0.5,
                        line=dict(color=color, width=2.5),
                        fillcolor="rgba(0,0,0,0)",
                        layer="above",
                    ))

                fig_cm = go.Figure(go.Heatmap(
                    z=z_plot, x=class_order, y=class_order,
                    colorscale="Blues", zmin=zmin, zmax=zmax,
                    text=text, texttemplate="%{text}", textfont={"size": 8},
                    hovertemplate="True: %{y}<br>Pred: %{x}<br>value: %{z:.3f}<extra></extra>",
                    colorbar=dict(title=cb_title),
                ))
                fig_cm.update_layout(
                    title=dict(text=f"Confusion matrix ({cm_mode.lower()}) — Epoch {selected_cm_ep}",
                               font=dict(size=13)),
                    xaxis=dict(title="Predicted", tickangle=45, tickfont=dict(size=9),
                               tickmode="array", tickvals=list(range(n_classes)),
                               ticktext=class_order),
                    yaxis=dict(title="True", tickfont=dict(size=9), autorange="reversed",
                               tickmode="array", tickvals=list(range(n_classes)),
                               ticktext=class_order),
                    height=660, margin=dict(l=180, r=20, t=50, b=180),
                    paper_bgcolor="white",
                    shapes=shapes,
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                # Group legend
                legend_html = " &nbsp; ".join(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{color};border-radius:2px;margin-right:4px;vertical-align:middle"></span>'
                    f'<span style="font-size:0.8rem">{name}</span>'
                    for name, (_, color) in _CLASS_GROUPS.items()
                )
                st.markdown(
                    f"<div style='margin-top:4px'>{legend_html}</div>"
                    "<div style='font-size:0.75rem;color:#64748b;margin-top:4px'>"
                    "Colored borders group classes by ecosystem type. "
                    "Diagonal = recall per class.</div>",
                    unsafe_allow_html=True,
                )

            elif run.confusion_matrix_paths:
                epoch_labels = [p.stem.split("_epoch")[-1] for p in run.confusion_matrix_paths]
                cm_idx = st.selectbox(
                    "Epoch", range(len(run.confusion_matrix_paths)),
                    format_func=lambda i: f"Epoch {epoch_labels[i]}",
                    key="cm_epoch_sel",
                )
                if run.confusion_matrix_paths[cm_idx].exists():
                    st.image(Image.open(run.confusion_matrix_paths[cm_idx]), use_container_width=True)
            else:
                st.info("No confusion matrix. Use `--layers confusion` to generate it.")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Batch Monitor
# ═══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    elif not run.batch_csv_path:
        st.info("No batch-level CSV for this run. Use `--layers batch-monitor` to generate it.")
    else:
        bdf = _load_batch(str(run.batch_csv_path))
        epochs_available = sorted(bdf["epoch"].unique())

        col_ep, col_ma = st.columns([3, 2])
        with col_ep:
            selected_epochs = st.multiselect(
                "Epochs", epochs_available, default=list(epochs_available[:3]),
            )
        with col_ma:
            ma_window = st.slider(
                "Moving average window (batches)", 0, 200, 20,
                help="0 = disabled",
            )

        if not bdf.empty:
            n_batches = int(bdf["n_batches"].iloc[0])
            st.caption(f"Batches per epoch: {n_batches}")

        if selected_epochs:
            fig = go.Figure()
            for i, ep in enumerate(selected_epochs):
                subset = bdf[bdf["epoch"] == ep].copy()
                color = COLORS[i % len(COLORS)]

                fig.add_trace(go.Scatter(
                    x=subset["batch"], y=subset["running_loss"],
                    name=f"Epoch {ep}", mode="lines",
                    line=dict(color=color, width=1),
                    opacity=0.4, legendgroup=f"ep{ep}",
                ))

                if ma_window > 0 and len(subset) >= ma_window:
                    ma = subset["running_loss"].rolling(ma_window, center=True).mean()
                    fig.add_trace(go.Scatter(
                        x=subset["batch"], y=ma,
                        name=f"Epoch {ep} MA{ma_window}", mode="lines",
                        line=dict(color=color, width=2.5),
                        legendgroup=f"ep{ep}",
                    ))
                    mean_l = subset["running_loss"].mean()
                    std_l = subset["running_loss"].std()
                    spikes = subset[subset["running_loss"] > mean_l + 2 * std_l]
                    if not spikes.empty:
                        fig.add_trace(go.Scatter(
                            x=spikes["batch"], y=spikes["running_loss"],
                            name=f"Spike Ep{ep}", mode="markers",
                            marker=dict(color="red", size=7, symbol="x"),
                            legendgroup=f"ep{ep}", showlegend=False,
                        ))

            fig.update_layout(
                **_base_layout(400, "Running loss per batch"),
                xaxis_title="Batch", yaxis_title="Running loss",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw data"):
                st.dataframe(bdf[bdf["epoch"].isin(selected_epochs)], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Compare Runs
# ═══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    if not runs:
        st.info("No runs available.")
    else:
        all_run_labels = {r.label: r for r in runs}
        all_labels_list = list(all_run_labels.keys())

        selected_compare = st.multiselect(
            "Select runs to compare (max 4)", all_labels_list,
            default=all_labels_list[:min(2, len(all_labels_list))],
            max_selections=4,
        )

        if len(selected_compare) < 2:
            st.info("Select at least 2 runs.")
        else:
            compare_runs = [(lbl, all_run_labels[lbl]) for lbl in selected_compare]
            compare_dfs: list[tuple[str, pd.DataFrame]] = []
            for lbl, r in compare_runs:
                cdf = _load_df(
                    str(r.log_path),
                    str(r.epoch_csv_path) if r.epoch_csv_path else None,
                )
                compare_dfs.append((lbl[:30], cdf))

            summary_rows = []
            for lbl, r in compare_runs:
                cdf = next(d for l, d in compare_dfs if l == lbl[:30])
                best_f1_c = _safe_max(cdf["val_f1"]) if ("val_f1" in cdf.columns and not cdf.empty) else float("nan")
                _best_ep_c_v = _safe_val_at_best(cdf, "val_f1", "epoch")
                best_ep_c = int(_best_ep_c_v) if _best_ep_c_v is not None else "—"
                _last = cdf["val_f1"].dropna() if ("val_f1" in cdf.columns and not cdf.empty) else pd.Series(dtype=float)
                final_f1_c = _last.iloc[-1] if not _last.empty else float("nan")
                total_s_c = cdf["epoch_time"].dropna().sum() if "epoch_time" in cdf.columns else float("nan")
                dur_c = (
                    f"{int(total_s_c//3600)}h {int((total_s_c%3600)//60)}m"
                    if not pd.isna(total_s_c) else "—"
                )
                summary_rows.append({
                    "Run": lbl[:50],
                    "Best Val F1": f"{best_f1_c:.4f}" if not pd.isna(best_f1_c) else "—",
                    "Best epoch": best_ep_c,
                    "Final F1": f"{final_f1_c:.4f}" if not pd.isna(final_f1_c) else "—",
                    "Epochs": len(cdf),
                    "Duration": dur_c,
                    "Env": r.env,
                    "Trace": r.trace_mode,
                })

            st.dataframe(pd.DataFrame(summary_rows).set_index("Run"), use_container_width=True)
            st.markdown("---")

            # ── Radar chart of best-epoch metrics ─────────────────────────────
            st.markdown("#### Best-epoch radar")
            radar_metrics = ["val_f1", "train_f1", "val_acc", "val_prec", "val_rec"]
            radar_fig = go.Figure()
            for i, (lbl, cdf) in enumerate(compare_dfs):
                vals = []
                for m_col in radar_metrics:
                    v = _safe_val_at_best(cdf, "val_f1", m_col)
                    vals.append(float(v) if v is not None else 0.0)
                vals_closed = vals + [vals[0]]
                cats_closed = radar_metrics + [radar_metrics[0]]
                radar_fig.add_trace(go.Scatterpolar(
                    r=vals_closed, theta=cats_closed,
                    fill="toself", name=lbl[:30],
                    line=dict(color=COLORS[i % len(COLORS)]),
                    opacity=0.6,
                ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=360,
                margin=dict(l=60, r=60, t=40, b=40),
                paper_bgcolor="white",
                title=dict(text="Metrics at best Val F1 epoch", font=dict(size=13)),
            )
            st.plotly_chart(radar_fig, use_container_width=True)
            st.markdown("---")

            metrics_to_compare = st.multiselect(
                "Metrics",
                ["val_f1", "val_loss", "train_f1", "train_loss", "val_prec", "val_rec", "epoch_time"],
                default=["val_f1", "val_loss"],
            )

            cols = st.columns(2)
            for idx, col_name in enumerate(metrics_to_compare):
                fig = _overlay_fig(compare_dfs, col=col_name,
                                   title=col_name.replace("_", " "), y_label=col_name)
                cols[idx % 2].plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Feasibility
# ═══════════════════════════════════════════════════════════════════════════════

with tab_feasibility:
    subtab_report, subtab_compare_feas, subtab_run_feas = st.tabs(
        ["Report", "Compare vs training", "Run check"]
    )

    with subtab_report:
        feasibility_csvs = _get_feasibility_csvs()

        if not feasibility_csvs:
            st.info("No feasibility CSVs found. Run the feasibility check to generate one.")
        else:
            csv_labels = {str(p): f"{p.parent.name}/{p.name}" for p in feasibility_csvs}
            selected_feas_path = st.selectbox(
                "Report", list(csv_labels.keys()),
                format_func=lambda p: csv_labels[p],
            )
            meta, bdf_feas = parse_feasibility_csv(Path(selected_feas_path))

            if meta:
                st.subheader("Model")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Model", meta.get("model_name", "—"))
                mc2.metric("Parameters (M)", meta.get("total_params_M", "—"))
                mc3.metric("FLOPs (MFLOPs)", meta.get("flops_mflops", "—"))
                mc4.metric("Hardware", meta.get("hardware_name", "—"))

                mem_keys = ["weight_mb", "gradient_mb", "optimizer_mb",
                            "activation_mb_per_image", "total_static_mb"]
                if any(k in meta for k in mem_keys):
                    st.subheader("Static memory")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Weights (MB)", meta.get("weight_mb", "—"))
                    m2.metric("Gradients (MB)", meta.get("gradient_mb", "—"))
                    m3.metric("AdamW state (MB)", meta.get("optimizer_mb", "—"))
                    m4.metric("Activations/img (MB)", meta.get("activation_mb_per_image", "—"))
                    m5.metric("Total static (MB)", meta.get("total_static_mb", "—"))

                st.subheader("Hardware")
                total_vram = meta.get("total_vram_gb")
                free_vram = meta.get("free_vram_gb")
                h1, h2, h3 = st.columns(3)
                h1.metric("Total VRAM (GB)", total_vram or "—")
                h2.metric("Free VRAM (GB)", free_vram or "—")
                if total_vram and free_vram:
                    pct = float(free_vram) / float(total_vram) * 100
                    h3.metric("Free VRAM %", f"{pct:.1f}%")
                    fig_vr = go.Figure(go.Bar(
                        x=["Free", "Used"],
                        y=[float(free_vram), float(total_vram) - float(free_vram)],
                        marker_color=[COLORS[2], COLORS[3]], opacity=0.85,
                    ))
                    fig_vr.update_layout(**_base_layout(200, "VRAM distribution"),
                                        yaxis_title="GB")
                    st.plotly_chart(fig_vr, use_container_width=True)

            if not bdf_feas.empty:
                st.subheader("Benchmark")
                st.dataframe(bdf_feas, use_container_width=True)

                viable = bdf_feas[bdf_feas["oom"] == "no"].copy()
                tp_col = _throughput_col(viable)

                if not viable.empty and tp_col:
                    # Throughput chart
                    has_split = ("imgs_per_s_train" in viable.columns
                                 and "imgs_per_s_eval" in viable.columns)
                    fig_tp = go.Figure()
                    for mode in viable["trace_mode"].unique():
                        sub = viable[viable["trace_mode"] == mode]
                        x_labels = sub["batch_size"].astype(str) + f" [{mode}]"
                        if has_split:
                            fig_tp.add_trace(go.Bar(x=x_labels, y=sub["imgs_per_s_train"],
                                                    name=f"Train [{mode}]"))
                            fig_tp.add_trace(go.Bar(x=x_labels, y=sub["imgs_per_s_eval"],
                                                    name=f"Eval [{mode}]"))
                        else:
                            fig_tp.add_trace(go.Bar(x=sub["batch_size"].astype(str),
                                                    y=sub[tp_col], name=f"trace={mode}"))
                    fig_tp.update_layout(
                        **_base_layout(320, "Throughput by batch size"),
                        barmode="group",
                        xaxis_title="Batch size", yaxis_title="imgs/s",
                    )
                    st.plotly_chart(fig_tp, use_container_width=True)

                    # VRAM chart
                    if "peak_vram_gb" in viable.columns and viable["peak_vram_gb"].notna().any():
                        fig_vram = go.Figure()
                        for mode in viable["trace_mode"].unique():
                            sub = viable[viable["trace_mode"] == mode]
                            fig_vram.add_trace(go.Bar(
                                x=sub["batch_size"].astype(str), y=sub["peak_vram_gb"],
                                name=f"trace={mode}",
                            ))
                        if meta and meta.get("free_vram_gb"):
                            fig_vram.add_hline(
                                y=float(meta["free_vram_gb"]),
                                line_dash="dash", line_color="red",
                                annotation_text=f"Free VRAM: {meta['free_vram_gb']} GB",
                                annotation_position="top left",
                            )
                        fig_vram.update_layout(
                            **_base_layout(280, "Peak VRAM by batch size"),
                            barmode="group",
                            xaxis_title="Batch size", yaxis_title="GB",
                        )
                        st.plotly_chart(fig_vram, use_container_width=True)

                # Estimates table
                est_cols = [c for c in bdf_feas.columns if c.startswith("est_")]
                if est_cols:
                    st.subheader("Time estimates")
                    orig_ep_col = next(
                        (c for c in bdf_feas.columns
                         if c.startswith("est_total_h_") and c.endswith("ep")), None
                    )
                    orig_n = None
                    if orig_ep_col:
                        try:
                            orig_n = int(orig_ep_col.split("est_total_h_")[1].replace("ep", ""))
                        except ValueError:
                            pass

                    recalc_n = st.number_input(
                        "Epochs for total estimate", min_value=1, value=orig_n or 30,
                    )
                    display_cols = ["batch_size", "trace_mode", "oom"]
                    for c in ["est_train_min_per_epoch", "est_eval_min_per_epoch",
                              "est_total_min_per_epoch", "est_min_per_epoch_30ep"]:
                        if c in bdf_feas.columns:
                            display_cols.append(c)
                    if orig_ep_col:
                        display_cols.append(orig_ep_col)

                    est_df = bdf_feas[[c for c in display_cols if c in bdf_feas.columns]].copy()

                    # Recalculate total hours
                    per_epoch_col = next(
                        (c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                         if c in bdf_feas.columns), None
                    )
                    if per_epoch_col:
                        est_df[f"est_total_h_{recalc_n}ep"] = (
                            bdf_feas[per_epoch_col] * recalc_n / 60
                        ).round(2)

                    st.dataframe(est_df, use_container_width=True)

    with subtab_compare_feas:
        st.markdown("### Feasibility estimates vs actual training results")
        st.caption(
            "Select a feasibility report and a training run with the same model "
            "to compare estimated vs measured values."
        )

        feasibility_csvs_cmp = _get_feasibility_csvs()
        all_runs_cmp = _get_runs()

        if not feasibility_csvs_cmp:
            st.info("No feasibility CSVs found.")
        elif not all_runs_cmp:
            st.info("No training runs found.")
        else:
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                csv_labels_cmp = {
                    str(p): f"{p.parent.name}/{p.name}" for p in feasibility_csvs_cmp
                }
                sel_feas_cmp = st.selectbox(
                    "Feasibility report",
                    list(csv_labels_cmp.keys()),
                    format_func=lambda p: csv_labels_cmp[p],
                    key="cmp_feas_sel",
                )
                meta_cmp, feas_df_cmp = parse_feasibility_csv(Path(sel_feas_cmp))
                model_feas = meta_cmp.get("model_name", "")

                batch_sizes_available = []
                if not feas_df_cmp.empty and "batch_size" in feas_df_cmp.columns:
                    batch_sizes_available = sorted(
                        feas_df_cmp["batch_size"].dropna().astype(int).unique().tolist()
                    )
                sel_bs = st.selectbox(
                    "Batch size", batch_sizes_available, key="cmp_bs_sel",
                ) if batch_sizes_available else None

                trace_modes_available = []
                if not feas_df_cmp.empty and "trace_mode" in feas_df_cmp.columns:
                    trace_modes_available = sorted(feas_df_cmp["trace_mode"].unique().tolist())
                sel_trace = st.selectbox(
                    "Trace mode", trace_modes_available or ["simple"], key="cmp_trace_sel",
                )

                nfs_factor_cmp = float(meta_cmp.get("nfs_factor", 1.0) or 1.0)
                st.caption(f"Model: **{model_feas or '—'}** | NFS factor: {nfs_factor_cmp:.2f}")

            with cmp_col2:
                run_labels_cmp = {r.label: r for r in all_runs_cmp}
                # Pre-filter to runs with same model if possible
                matching = [
                    lbl for lbl, r in run_labels_cmp.items()
                    if model_feas and r.model and model_feas in r.model
                ]
                default_run = matching[0] if matching else list(run_labels_cmp.keys())[0]
                sel_run_cmp = st.selectbox(
                    "Training run",
                    list(run_labels_cmp.keys()),
                    index=list(run_labels_cmp.keys()).index(default_run),
                    key="cmp_run_sel",
                )
                run_cmp = run_labels_cmp[sel_run_cmp]
                actual_df_cmp = _load_df(
                    str(run_cmp.log_path),
                    str(run_cmp.epoch_csv_path) if run_cmp.epoch_csv_path else None,
                )

            if sel_bs is not None and not actual_df_cmp.empty:
                comparison = build_comparison(
                    meta=meta_cmp,
                    feas_df=feas_df_cmp,
                    actual_df=actual_df_cmp,
                    batch_size=int(sel_bs),
                    trace_mode=sel_trace,
                    nfs_factor=nfs_factor_cmp,
                )
                if comparison:
                    cmp_table = comparison.to_dataframe()

                    def _color_error(val: str) -> str:
                        try:
                            v = float(val.replace("%", "").replace("+", ""))
                            if abs(v) <= 10:
                                return "background-color: #dcfce7"
                            if abs(v) <= 30:
                                return "background-color: #fef9c3"
                            return "background-color: #fee2e2"
                        except (ValueError, AttributeError):
                            return ""

                    err_col = next(
                        (c for c in cmp_table.columns if c == "Error %"), None
                    )
                    styled_cmp = cmp_table.style
                    if err_col:
                        styled_cmp = styled_cmp.map(_color_error, subset=[err_col])
                    st.dataframe(styled_cmp, use_container_width=True, hide_index=True)

                    st.caption(
                        "Green = error ≤ 10% | Yellow = 10–30% | Red = > 30%. "
                        "Feasibility uses synthetic batches (no real I/O) — "
                        "NFS and data loading overhead explain most of the gap."
                    )

                    with st.expander("Interpretation guide"):
                        st.markdown("""
**Train time / epoch**: Estimated as `n_batches × s/batch × nfs_factor`.
The synthetic benchmark does not read real images from disk, so NFS I/O is the
main source of underestimation. Multiply by `nfs_factor` (1.3 for Verode) to compensate.

**Eval time / epoch**: Estimated as `n_val_batches × s/batch_eval`.
No NFS factor applied (eval is smaller and less I/O-bound in practice).

**Train throughput (imgs/s)**: Measured on synthetic batches in GPU memory.
Real throughput is lower due to data loading latency.

**Peak VRAM**: Estimated as `static_mem + batch_size × activation_mb_per_img`.
The formula gives a conservative upper bound; PyTorch's allocator is more efficient.

**FLOPs / train epoch**: `FLOPs/image × N_train_images`.
Theoretical compute — does not account for backward pass overhead (~2× forward).
                        """)
                else:
                    st.warning(
                        f"No matching feasibility row for batch_size={sel_bs}, "
                        f"trace_mode={sel_trace}."
                    )

    with subtab_run_feas:
        st.subheader("Run feasibility check")
        configs_available = _get_configs()
        model_options_f = [
            "vit_base_patch16_224", "vit_tiny_patch16_224",
            "vit_small_patch16_224", "resnet50", "efficientnet_b0",
        ]
        with st.form("feasibility_form"):
            fa1, fa2 = st.columns(2)
            with fa1:
                feas_model = st.selectbox("Model", model_options_f)
                feas_batches = st.multiselect(
                    "Batch sizes", [16, 32, 64, 128], default=[32, 64],
                )
                feas_epochs = st.number_input("Epochs for estimate", min_value=1, value=30)
            with fa2:
                feas_traces = st.multiselect(
                    "Trace modes", ["off", "simple", "deep"], default=["off", "simple"],
                )
                feas_nfs = st.slider("NFS factor", 1.0, 2.0, 1.0, 0.05,
                                     help="Correction for NFS latency (Verode: ~1.3)")
                feas_config = st.selectbox(
                    "Config YAML (optional)",
                    ["(none)"] + (configs_available if configs_available else []),
                )
            submitted_feas = st.form_submit_button("Run")

        if submitted_feas:
            if not feas_batches:
                st.error("Select at least one batch size.")
            else:
                bs_args = " ".join(str(b) for b in feas_batches)
                trace_args = " ".join(feas_traces) if feas_traces else "off"
                parts = [
                    "uv run python scripts/check_feasibility.py",
                    f"--model {feas_model}",
                    f"--batch-sizes {bs_args}",
                    f"--epochs {feas_epochs}",
                    f"--trace-modes {trace_args}",
                ]
                if feas_nfs != 1.0:
                    parts.append(f"--nfs-factor {feas_nfs}")
                if feas_config != "(none)":
                    parts.append(f"--config configs/{feas_config}")
                cmd = " ".join(parts)
                st.code(cmd, language="bash")
                out_ph = st.empty()
                with st.spinner("Running…"):
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, cwd=str(ROOT),
                    )
                if result.returncode == 0:
                    st.success("Done.")
                    out_ph.code(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
                    _get_feasibility_csvs.clear()
                else:
                    st.error("Error:")
                    out_ph.code(result.stderr[-2000:])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Time Analysis
# ═══════════════════════════════════════════════════════════════════════════════

with tab_time:
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
        df_time = _load_df(
            str(run.log_path),
            str(run.epoch_csv_path) if run.epoch_csv_path else None,
        )

        if "epoch_time" not in df_time.columns or df_time["epoch_time"].isna().all():
            st.info("No epoch time data. Use `--trace simple` to generate it.")
        else:
            et = df_time[["epoch", "epoch_time"]].dropna()
            total_s = et["epoch_time"].sum()
            avg_s = et["epoch_time"].mean()

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Total", f"{int(total_s//3600)}h {int((total_s%3600)//60)}m")
            t2.metric("Avg/epoch", f"{avg_s/60:.1f} min")
            t3.metric("Min/epoch", f"{et['epoch_time'].min()/60:.1f} min")
            t4.metric("Max/epoch", f"{et['epoch_time'].max()/60:.1f} min")

            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=et["epoch"], y=et["epoch_time"] / 60,
                name="Real (min)", mode="lines+markers",
                line=dict(color=COLORS[0], width=2), marker=dict(size=4),
            ))

            has_train_t = ("epoch_time_train_s" in df_time.columns
                           and df_time["epoch_time_train_s"].notna().any())
            has_eval_t = ("epoch_time_eval_s" in df_time.columns
                          and df_time["epoch_time_eval_s"].notna().any())
            if has_train_t:
                fig_time.add_trace(go.Scatter(
                    x=et["epoch"], y=df_time["epoch_time_train_s"] / 60,
                    name="Train (min)", mode="lines",
                    line=dict(color=COLORS[2], width=2, dash="dot"),
                ))
            if has_eval_t:
                fig_time.add_trace(go.Scatter(
                    x=et["epoch"], y=df_time["epoch_time_eval_s"] / 60,
                    name="Eval (min)", mode="lines",
                    line=dict(color=COLORS[1], width=2, dash="dash"),
                ))

            if len(et) >= 2:
                x_arr = et["epoch"].values.astype(float)
                y_arr = et["epoch_time"].values / 60
                coeffs = np.polyfit(x_arr, y_arr, 1)
                fig_time.add_trace(go.Scatter(
                    x=et["epoch"], y=np.polyval(coeffs, x_arr),
                    name="Trend", mode="lines",
                    line=dict(color="#94a3b8", width=1, dash="dash"),
                ))

            # Warmup shading
            warmup_ep = None
            for cfg_name in _get_configs():
                try:
                    import yaml
                    cfg = yaml.safe_load((ROOT / "configs" / cfg_name).read_text())
                    warmup_ep = cfg.get("training", {}).get("warmup_epochs")
                    if warmup_ep:
                        break
                except Exception:
                    pass
            if warmup_ep:
                fig_time.add_vrect(
                    x0=0.5, x1=warmup_ep + 0.5,
                    fillcolor="#f59e0b", opacity=0.07,
                    annotation_text=f"Warmup ({warmup_ep} ep)",
                    annotation_position="top left",
                )

            # Feasibility overlay
            feasibility_csvs = _get_feasibility_csvs()
            if feasibility_csvs:
                try:
                    _, bdf_t = parse_feasibility_csv(feasibility_csvs[0])
                    viable_t = bdf_t[bdf_t["oom"] == "no"].copy()
                    tp_col_t = _throughput_col(viable_t)
                    per_ep_col = next(
                        (c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                         if c in viable_t.columns), None
                    )
                    _idx_t = viable_t[tp_col_t].idxmax() if (tp_col_t and per_ep_col and not viable_t.empty) else None
                    if _idx_t is not None and not pd.isna(_idx_t):
                        best_row = viable_t.loc[_idx_t]
                        est_min = float(best_row[per_ep_col])
                        fig_time.add_hline(
                            y=est_min, line_dash="dash", line_color=COLORS[1],
                            annotation_text=f"Feasibility estimate: {est_min:.0f} min/epoch",
                            annotation_position="top right",
                        )
                except Exception:
                    pass

            fig_time.update_layout(
                **_base_layout(380, "Time per epoch"),
                xaxis_title="Epoch", yaxis_title="Minutes",
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # Estimated vs real
            if feasibility_csvs:
                try:
                    _, bdf_c = parse_feasibility_csv(feasibility_csvs[0])
                    viable_c = bdf_c[bdf_c["oom"] == "no"].copy()
                    tp_col_c = _throughput_col(viable_c)
                    per_ep_c = next(
                        (c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                         if c in viable_c.columns), None
                    )
                    _idx_c = viable_c[tp_col_c].idxmax() if (tp_col_c and per_ep_c and not viable_c.empty) else None
                    if _idx_c is not None and not pd.isna(_idx_c):
                        best_c = viable_c.loc[_idx_c]
                        est_val = float(best_c[per_ep_c])
                        real_val = avg_s / 60
                        err_pct = (real_val - est_val) / est_val * 100 if est_val else 0
                        st.markdown("**Estimated vs Real**")
                        ce1, ce2, ce3 = st.columns(3)
                        ce1.metric("Estimated (min/epoch)", f"{est_val:.1f}")
                        ce2.metric("Actual avg (min/epoch)", f"{real_val:.1f}")
                        ce3.metric("Relative error", f"{err_pct:+.1f}%")
                except Exception:
                    pass

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 7 — Run Info
# ═══════════════════════════════════════════════════════════════════════════════

with tab_info:
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
        df_info = _load_df(
            str(run.log_path),
            str(run.epoch_csv_path) if run.epoch_csv_path else None,
        )
        n_ep_i = len(df_info)
        best_f1_i = _safe_max(df_info["val_f1"]) if "val_f1" in df_info.columns else float("nan")
        _best_ep_i_v = _safe_val_at_best(df_info, "val_f1", "epoch")
        best_ep_i = int(_best_ep_i_v) if _best_ep_i_v is not None else "—"

        col_m, col_f = st.columns(2)

        with col_m:
            st.subheader("Run metadata")
            rows_i = {
                "Log": run.log_path.name,
                "Env": run.env,
                "Trace mode": run.trace_mode,
                "Epochs": n_ep_i,
                "Best Val F1": f"{best_f1_i:.4f}" if not pd.isna(best_f1_i) else "—",
                "Best epoch": best_ep_i,
            }
            if "epoch_time" in df_info.columns and df_info["epoch_time"].notna().any():
                total_si = df_info["epoch_time"].sum()
                rows_i["Total time"] = f"{int(total_si//3600)}h {int((total_si%3600)//60)}m"
                rows_i["Avg/epoch"] = f"{df_info['epoch_time'].mean()/60:.1f} min"
            for k, v in rows_i.items():
                st.markdown(f"**{k}:** {v}")

        with col_f:
            st.subheader("Associated files")
            for label, path in [
                ("Plot", run.plot_path),
                ("Batch CSV", run.batch_csv_path),
                ("Per-class CSV", run.perclass_csv_path),
                ("Epoch CSV", run.epoch_csv_path),
            ]:
                st.markdown(f"- **{label}:** `{path.name if path else '—'}`")
            for p in run.perclass_paths:
                st.markdown(f"- Per-class PNG: `{p.name}`")
            for p in run.confusion_matrix_paths:
                st.markdown(f"- Confusion matrix PNG: `{p.name}`")

        st.markdown("---")

        # Config YAML
        st.subheader("Config YAML")
        import yaml
        configs_i: list[Path] = []
        for cfg in _get_configs():
            try:
                cfg_path = ROOT / "configs" / cfg
                cfg_data = yaml.safe_load(cfg_path.read_text())
                env_cfg = cfg_data.get("output", {}).get("env", "")
                if env_cfg == run.env or (run.env == "local" and "cluster" not in cfg):
                    configs_i.append(cfg_path)
            except Exception:
                pass
        if configs_i:
            cfg_sel = st.selectbox("Config", [p.name for p in configs_i])
            cfg_path_sel = next(p for p in configs_i if p.name == cfg_sel)
            st.code(cfg_path_sel.read_text(), language="yaml")
        else:
            st.caption("Could not determine config for this run.")

        # Anomalies
        st.subheader("Anomaly scan")
        anomalies = _detect_anomalies(run.log_path)
        if anomalies:
            st.warning(f"{len(anomalies)} anomaly lines detected.")
            with st.expander("View anomalies"):
                for line in anomalies:
                    st.text(line)
        else:
            st.success("No anomalies detected in log.")

        # Full searchable log
        st.subheader("Log")
        search_term = st.text_input("Filter log lines", "")
        try:
            all_lines = run.log_path.read_text(errors="replace").splitlines()
            if search_term:
                disp_lines = [l for l in all_lines if search_term.lower() in l.lower()]
                st.caption(f"{len(disp_lines)} / {len(all_lines)} lines")
            else:
                disp_lines = all_lines
                st.caption(f"{len(all_lines)} lines total")
            st.code("\n".join(disp_lines[-400:]), language="text")
        except Exception as exc:
            st.error(str(exc))

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 8 — Launcher
# ═══════════════════════════════════════════════════════════════════════════════

with tab_launcher:
    subtab_single, subtab_ddp = st.tabs(["Single GPU", "DDP (multi-GPU)"])

    configs_l = _get_configs()
    model_opts_l = [
        "vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224",
        "resnet50", "efficientnet_b0", "deit_tiny_patch16_224",
    ]

    with subtab_single:
        st.subheader("Single GPU training")
        with st.form("launcher_single_form"):
            la1, la2 = st.columns(2)
            with la1:
                l_model = st.selectbox("Model", model_opts_l)
                l_config = st.selectbox("Config YAML", configs_l if configs_l else ["(none)"])
                l_epochs = st.number_input("Epochs override", min_value=0, value=0,
                                           help="0 = use config value")
                l_batch = st.number_input("Batch size override", min_value=0, value=0,
                                          help="0 = use config value")
            with la2:
                l_trace = st.selectbox("Trace mode", ["simple", "off", "deep"])
                l_layers = st.multiselect(
                    "Layers", ["plot", "hooks", "confusion", "batch-monitor"],
                    default=["plot", "confusion"],
                )
                l_fn = st.multiselect("Fn decorators", ["timing", "energy"])
                l_inspect = st.multiselect(
                    "Inspect features",
                    ["model-summary", "grad-monitor", "anomalies", "batch-table"],
                )
            launched_single = st.form_submit_button("Launch")

        if launched_single:
            parts_l = [
                "uv run python scripts/train_single_gpu.py",
                f"--config configs/{l_config}",
                f"--model {l_model}",
                f"--trace {l_trace}",
            ]
            if l_epochs > 0:
                parts_l.append(f"--epochs {l_epochs}")
            if l_batch > 0:
                parts_l.append(f"--batch-size {l_batch}")
            if l_layers:
                parts_l.append(f"--layers {' '.join(l_layers)}")
            if l_fn:
                parts_l.append(f"--fn {' '.join(l_fn)}")
            if l_inspect:
                parts_l.append(f"--inspect {' '.join(l_inspect)}")
            cmd_l = " ".join(parts_l)
            st.code(cmd_l, language="bash")
            out_ph_l = st.empty()
            rc_l = _launch_process(cmd_l, out_ph_l)
            if rc_l == 0:
                st.success("Training complete.")
                _get_runs.clear()
            else:
                st.error(f"Process exited with code {rc_l}.")

    with subtab_ddp:
        st.subheader("DDP training")
        with st.form("launcher_ddp_form"):
            dd1, dd2 = st.columns(2)
            with dd1:
                d_nproc = st.number_input("GPUs (--nproc_per_node)", min_value=1, max_value=8, value=2)
                d_model = st.selectbox("Model", model_opts_l, key="ddp_model")
                d_config = st.selectbox(
                    "Config YAML", configs_l if configs_l else ["(none)"], key="ddp_config",
                )
                d_epochs = st.number_input("Epochs override", min_value=0, value=0, key="ddp_ep")
            with dd2:
                d_trace = st.selectbox("Trace mode", ["simple", "off", "deep"], key="ddp_trace")
                d_layers = st.multiselect(
                    "Layers", ["plot", "confusion", "batch-monitor"],
                    default=["plot"], key="ddp_layers",
                )
                d_fn = st.multiselect("Fn decorators", ["timing", "energy"], key="ddp_fn")
            launched_ddp = st.form_submit_button("Launch")

        if launched_ddp:
            parts_d = [
                f"torchrun --nproc_per_node={d_nproc} scripts/train_ddp.py",
                f"--config configs/{d_config}",
                f"--model {d_model}",
                f"--trace {d_trace}",
            ]
            if d_epochs > 0:
                parts_d.append(f"--epochs {d_epochs}")
            if d_layers:
                parts_d.append(f"--layers {' '.join(d_layers)}")
            if d_fn:
                parts_d.append(f"--fn {' '.join(d_fn)}")
            cmd_d = " ".join(parts_d)
            st.code(cmd_d, language="bash")
            out_ph_d = st.empty()
            rc_d = _launch_process(cmd_d, out_ph_d)
            if rc_d == 0:
                st.success("DDP training complete.")
                _get_runs.clear()
            else:
                st.error(f"Process exited with code {rc_d}.")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 9 — Live Monitor
# ═══════════════════════════════════════════════════════════════════════════════

with tab_live:
    st.subheader("Live Monitor")

    now_ts = time.time()
    recent_runs = [
        r for r in runs
        if r.log_path.exists() and (now_ts - r.log_path.stat().st_mtime) < 1800
    ]

    if not recent_runs:
        st.info(
            "No active runs (no log modified in the last 30 min). "
            "Launch a training run from the Launcher tab."
        )
    else:
        live_labels = {r.label: r for r in recent_runs}
        live_run = live_labels[
            st.selectbox("Active run", list(live_labels.keys()), key="live_run_sel")
        ]

        gpu = _gpu_usage()
        if gpu:
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("GPU", gpu["name"])
            g2.metric(
                "VRAM",
                f"{gpu['mem_used_mb']/1024:.1f} / {gpu['mem_total_mb']/1024:.1f} GB",
            )
            g3.metric("Utilization", f"{gpu['util_pct']}%")
            g4.metric("Temperature", f"{gpu['temp_c']} °C")
        else:
            st.caption("GPU info unavailable (nvidia-smi not found).")

        progress = _parse_log_progress(live_run.log_path)
        if progress["epochs"] > 0:
            pct = progress["epoch"] / progress["epochs"]
            st.progress(pct, text=f"Epoch {progress['epoch']} / {progress['epochs']}")

        if progress["last_val_f1"] is not None:
            m1, m2 = st.columns(2)
            m1.metric("Last Val F1", f"{progress['last_val_f1']:.4f}")
            if progress["last_val_loss"] is not None:
                m2.metric("Last Val Loss", f"{progress['last_val_loss']:.4f}")

        if live_run.epoch_csv_path and live_run.epoch_csv_path.exists():
            live_df = _load_df(str(live_run.log_path), str(live_run.epoch_csv_path))
            if not live_df.empty:
                fig_live = go.Figure()
                if "val_f1" in live_df.columns:
                    fig_live.add_trace(go.Scatter(
                        x=live_df["epoch"], y=live_df["val_f1"],
                        name="Val F1", mode="lines+markers",
                        line=dict(color=COLORS[0], width=2), marker=dict(size=4),
                    ))
                if "val_loss" in live_df.columns:
                    fig_live.add_trace(go.Scatter(
                        x=live_df["epoch"], y=live_df["val_loss"],
                        name="Val Loss", mode="lines+markers",
                        line=dict(color=COLORS[1], width=2), marker=dict(size=4),
                    ))
                fig_live.update_layout(
                    **_base_layout(280, "Metrics"),
                    xaxis_title="Epoch",
                )
                st.plotly_chart(fig_live, use_container_width=True)

        st.subheader("Log tail")
        st.code(_read_log_tail(live_run.log_path, n=40), language="text")

    if live_mode:
        time.sleep(refresh_interval)
        _load_df.clear()
        st.rerun()
