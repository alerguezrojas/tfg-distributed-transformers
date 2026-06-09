"""Streamlit web dashboard — Training Dashboard v6 (English)."""

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

from src.web.batch_parser import parse_batch_csv
from src.web.confusion_matrix_parser import get_matrix_for_epoch, parse_confusion_matrix_csv
from src.web.dataset_stats import (
    CLASS_NAMES, SPLIT_SIZES,
    class_distribution_approximate, class_distribution_from_parquet,
    get_country_distribution, find_example_patches, load_rgb_image,
)
from src.web.feasibility_comparison import build_comparison
from src.web.feasibility_parser import parse_feasibility_csv, parse_ddp_scenarios
from src.web.log_parser import parse_log
from src.web.model_explorer import ALL_FAMILIES, CURATED_MODELS, compare_models
from src.web.perclass_parser import parse_perclass_csv
from src.web.run_registry import RunInfo, discover_feasibility_csvs, discover_runs
from src.web.system_monitor import get_snapshot

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#64748b", "#ec4899", "#94a3b8"]

# ── Page configuration ──────────────────────────────────────────────────────────

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
  [data-baseweb="tab-list"] {
    overflow-x: auto !important; flex-wrap: nowrap !important;
    scrollbar-width: thin; gap: 0 !important;
  }
  [data-baseweb="tab-list"]::-webkit-scrollbar { height: 3px; }
  [data-baseweb="tab-list"]::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
  [data-baseweb="tab"] {
    white-space: nowrap !important; font-size: 0.82rem !important;
    padding-left: 0.75rem !important; padding-right: 0.75rem !important;
    min-width: unset !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_df(log_path: str, epoch_csv: str | None) -> pd.DataFrame:
    if epoch_csv and Path(epoch_csv).exists():
        df = pd.read_csv(epoch_csv)
        if not df.empty:
            if "epoch_time_s" in df.columns:
                df = df.rename(columns={"epoch_time_s": "epoch_time"})
            # Energy/power and timings only live in the log (they are not
            # written to the epoch_metrics CSV). If the log exists, we merge
            # them by epoch so the energy panel also shows up with the CSV.
            _energy_cols = ["energy_train_j", "energy_eval_j", "energy_eval_wh",
                            "power_train_w", "power_eval_w", "time_train_s", "time_eval_s"]
            missing = [c for c in _energy_cols if c not in df.columns
                       or not df[c].notna().any()]
            if missing and log_path and Path(log_path).exists():
                log_df = parse_log(Path(log_path))
                merge_cols = [c for c in missing if c in log_df.columns]
                if merge_cols and "epoch" in log_df.columns:
                    df = df.merge(
                        log_df[["epoch", *merge_cols]], on="epoch", how="left"
                    )
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


@st.cache_data(ttl=60)
def _feas_label(path_str: str) -> str:
    """Readable label for a feasibility CSV: 'env · model · DD/MM HH:MM'
    instead of the raw date-based filename."""
    import re
    p = Path(path_str)
    env = p.parent.parent.name if p.parent.parent else "?"
    try:
        m, _ = parse_feasibility_csv(p)
        model = str(m.get("model_name", "?")).replace("_patch16_224", "")
    except Exception:
        model = "?"
    mt = re.search(r"(\d{2})(\d{2})\d{4}_(\d{2})(\d{2})", p.name)
    when = f"{mt.group(1)}/{mt.group(2)} {mt.group(3)}:{mt.group(4)}" if mt else p.stem
    return f"{env} · {model} · {when}"


@st.cache_data(ttl=30)
def _run_config(log_path_str: str) -> dict:
    """Extracts the 'Configuración: k=v | k=v | ...' line from the log → dict.
    Returns {} if the run predates this version (it does not record it).
    Note: the log key stays 'Configuración:' to match existing/backfilled logs."""
    try:
        for line in Path(log_path_str).read_text(errors="replace").splitlines():
            i = line.find("Configuración:")
            if i < 0:
                continue
            out = {}
            for part in line[i + len("Configuración:"):].split("|"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    out[k.strip()] = v.strip()
            return out
    except Exception:
        pass
    return {}


# ── Cached dataset loaders ──────────────────────────────────────────────────────


@st.cache_data(ttl=600)
def _load_class_distribution(parquet_str: str) -> pd.DataFrame | None:
    """Cached class distribution (iterates ~237K rows, slow)."""
    return class_distribution_from_parquet(Path(parquet_str))


@st.cache_data(ttl=600)
def _load_example_images(parquet_str: str, root_str: str, class_name: str, n: int = 4):
    """Loads n example RGB images for a class, cached."""
    patches = find_example_patches(Path(parquet_str), class_name, n=n)
    images = []
    for pid in patches:
        img = load_rgb_image(Path(root_str), pid)
        if img is not None:
            images.append((pid, img))
    return images


# ── General helpers ─────────────────────────────────────────────────────────────


def _safe_max(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.max()) if not valid.empty else float("nan")


def _safe_idxmax(series: pd.Series):
    valid = series.dropna()
    return valid.idxmax() if not valid.empty else None


def _safe_val_at_best(df: pd.DataFrame, metric_col: str, target_col: str):
    if metric_col not in df.columns or target_col not in df.columns:
        return None
    idx = _safe_idxmax(df[metric_col])
    if idx is None:
        return None
    v = df.loc[idx, target_col]
    return None if pd.isna(v) else v


def _throughput_col(df: pd.DataFrame) -> str | None:
    for col in ("imgs_per_s_train", "imgs_per_s"):
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _dur_str(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"


def _get_configs() -> list[str]:
    cfg_dir = ROOT / "configs"
    if not cfg_dir.exists():
        return []
    return sorted(p.name for p in cfg_dir.glob("*.yaml"))


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


def _read_log_tail(log_path: Path, n: int = 40) -> str:
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def _parse_log_progress(log_path: Path) -> dict:
    import re
    result = {"epoch": 0, "epochs": 0, "last_val_f1": None, "last_val_loss": None}
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        for line in reversed(lines):
            if "Epoch" in line and "/" in line:
                mm = re.search(r"Epoch\s+(\d+)/(\d+)", line)
                if mm:
                    result["epoch"] = int(mm.group(1))
                    result["epochs"] = int(mm.group(2))
                    break
        for line in reversed(lines):
            if "val_f1" in line or "val=0." in line:
                mm = re.search(r"val_f1[=\s]+([\d.]+)", line)
                if mm:
                    result["last_val_f1"] = float(mm.group(1))
                mm2 = re.search(r"val_loss[=\s]+([\d.]+)", line)
                if mm2:
                    result["last_val_loss"] = float(mm2.group(1))
                break
    except Exception:
        pass
    return result


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


# ── Chart helpers ───────────────────────────────────────────────────────────────

_PLOTLY_CFG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def _show(fig: go.Figure, key: str | None = None) -> None:
    """Shows a Plotly chart with a visible toolbar and PNG download."""
    cfg = dict(_PLOTLY_CFG)
    if key:
        cfg["toImageButtonOptions"] = {"format": "png", "scale": 2, "filename": key}
    st.plotly_chart(fig, use_container_width=True, config=cfg)


def _dl_csv(df: pd.DataFrame, filename: str = "data.csv", label: str = "Download CSV") -> None:
    """Download button for a DataFrame as CSV."""
    st.download_button(label, df.to_csv(index=False).encode(), file_name=filename, mime="text/csv")


def _base_layout(height: int = 320, title: str = "", margin: dict | None = None) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13)),
        height=height,
        margin=margin if margin is not None else dict(l=50, r=16, t=36, b=40),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
    )


def _metric_fig(
    df: pd.DataFrame,
    col_train: str, col_val: str,
    title: str, y_label: str,
    color_train: str = COLORS[0], color_val: str = COLORS[1],
    extra_traces: list | None = None,
    height: int = 320,
) -> go.Figure:
    fig = go.Figure()
    if col_train in df.columns and df[col_train].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_train], name="Train",
            mode="lines+markers", line=dict(color=color_train, width=2), marker=dict(size=4),
        ))
    if col_val in df.columns and df[col_val].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_val], name="Val",
            mode="lines+markers", line=dict(color=color_val, width=2), marker=dict(size=4),
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
    col: str, title: str, y_label: str,
    height: int = 340,
) -> go.Figure:
    fig = go.Figure()
    for i, (label, df) in enumerate(dfs):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df[col],
                name=label[:30], mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2), marker=dict(size=4),
            ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Epoch", yaxis_title=y_label,
        height=height, margin=dict(l=50, r=16, t=36, b=40),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
    )
    return fig


# ── Class groups (for the confusion matrix) ──────────────────────────────────

_CLASS_GROUPS = {
    "Urban":        ([0, 1],             "#6b7280"),
    "Agricultural": ([2, 3, 4, 5, 6, 7], "#d97706"),
    "Forest":       ([8, 9, 10, 13],     "#16a34a"),
    "Scrub/grass":  ([11, 12],           "#84cc16"),
    "Bare/coastal": ([14],               "#92400e"),
    "Wetlands":     ([15, 16],           "#0891b2"),
    "Water":        ([17, 18],           "#1d4ed8"),
}
_CLASS_GROUP_COLOR: dict[int, str] = {
    idx: color for name, (idxs, color) in _CLASS_GROUPS.items() for idx in idxs
}

# ── Sidebar ────────────────────────────────────────────────────────────────────

runs = _get_runs()

with st.sidebar:
    st.markdown("### Training Dashboard")
    st.markdown("---")

    if not runs:
        st.warning("No runs found in logs/.")
        selected_run = None
        run = None
    else:
        trace_filter = st.selectbox("Trace mode", ["all", "simple", "deep"])
        filtered = [r for r in runs if trace_filter == "all" or r.trace_mode == trace_filter]

        if not filtered:
            st.warning("No runs match this filter.")
            selected_run = None
            run = None
        else:
            run_labels = {r.label: r for r in filtered}
            selected_label = st.selectbox("Run", list(run_labels.keys()))
            run = run_labels[selected_label]
            selected_run = run

            st.markdown("---")
            has_csv = run.epoch_csv_path is not None and run.epoch_csv_path.exists()
            st.caption(
                f"**Log:** {run.log_path.name}  \n"
                f"**Environment:** {run.env}  \n"
                f"**Mode:** {run.mode}  \n"
                f"**Model:** {run.model or '—'}  \n"
                f"**Trace:** {run.trace_mode}  \n"
                f"**Epoch CSV:** {'yes' if has_csv else 'no'}  \n"
                f"**Batch CSV:** {'yes' if run.batch_csv_path else 'no'}  \n"
                f"**Per-class CSV:** {'yes' if run.perclass_csv_path else 'no'}"
            )

    st.markdown("---")
    st.markdown("**Live monitor**")
    refresh_interval = st.slider("Refresh interval (s)", 5, 60, 10)

# ── Tabs ────────────────────────────────────────────────────────────────────────

# 6 top-level tabs. The former 14 are nested as sub-tabs under these parents.
# Streamlit places each container where it is CREATED, so the `with tab_X:`
# blocks further down (untouched) fill these sub-tabs in place.
(
    tab_inicio, tab_run, tab_comp, tab_viabilidad, tab_datos, tab_sistema,
) = st.tabs([
    "Home", "Run", "Comparison", "Feasibility", "Data & models", "System",
])

with tab_run:
    st.caption("Details of the run selected in the sidebar.")
    tab_curvas, tab_porclase, tab_batch, tab_tiempo, tab_info = st.tabs(
        ["Curves", "Per-class", "Batch", "Time", "Info"])

with tab_comp:
    tab_ddp, tab_comparar = st.tabs(["Single vs Distributed", "Overlay runs"])

with tab_datos:
    tab_dataset, tab_modelos = st.tabs(["Dataset", "Models"])

with tab_sistema:
    tab_monitor, tab_envivo, tab_lanzador = st.tabs(["Monitor", "Live", "Launcher"])

# ═══════════════════════════════════════════════════════════════════════════════
# HOME — main screen with summary grid
# ═══════════════════════════════════════════════════════════════════════════════

with tab_inicio:
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

    # ── System status ───────────────────────────────────────────────────────────
    st.markdown("### System status")
    gpu_home = _gpu_usage()

    if gpu_home:
        # Full GPU name as a heading (avoids truncation in st.metric)
        gpu_name_clean = gpu_home["name"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        st.markdown(f"**GPU:** {gpu_home['name']}")
        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Model", gpu_name_clean)
        gc2.metric("VRAM", f"{gpu_home['mem_used_mb']/1024:.1f} / {gpu_home['mem_total_mb']/1024:.1f} GB")
        gc3.metric("GPU utilization", f"{gpu_home['util_pct']}%")
        gc4.metric("Temperature", f"{gpu_home['temp_c']} °C")
    else:
        st.caption("GPU: nvidia-smi unavailable")

    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        sc1, sc2 = st.columns(2)
        sc1.metric("CPU", f"{cpu_pct:.1f}%")
        sc2.metric("RAM", f"{ram.used/1024**3:.1f} / {ram.total/1024**3:.1f} GB  ({ram.percent:.0f}%)")
    except Exception:
        pass

    st.markdown("---")

    # ── Per-class snapshot ──────────────────────────────────────────────────────
    perclass_csv_home: Path | None = None
    if selected_run is not None and selected_run.perclass_csv_path and selected_run.perclass_csv_path.exists():
        perclass_csv_home = selected_run.perclass_csv_path
    else:
        all_pc = sorted(ROOT.rglob("perclass_metrics_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if all_pc:
            perclass_csv_home = all_pc[0]

    if perclass_csv_home is not None:
        try:
            pc_home = parse_perclass_csv(perclass_csv_home)
            if not pc_home.empty:
                last_ep_home = pc_home["epoch"].max()
                ep_pc_home = pc_home[pc_home["epoch"] == last_ep_home].copy().sort_values("f1", ascending=False)
                st.markdown(f"### Per-class performance (epoch {last_ep_home})")
                ph_left, ph_right = st.columns(2)
                with ph_left:
                    st.markdown("**Top 5 best classes**")
                    top5 = ep_pc_home.head(5)[["class_name", "f1", "precision", "recall"]]
                    st.dataframe(
                        top5.style.map(_color_f1_cell, subset=["f1"])
                            .format({"f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"}),
                        use_container_width=True, hide_index=True,
                    )
                with ph_right:
                    st.markdown("**Top 5 worst classes**")
                    bot5 = ep_pc_home.tail(5).sort_values("f1")[["class_name", "f1", "precision", "recall"]]
                    st.dataframe(
                        bot5.style.map(_color_f1_cell, subset=["f1"])
                            .format({"f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"}),
                        use_container_width=True, hide_index=True,
                    )
                st.markdown("---")
        except Exception:
            pass

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

# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA
# ═══════════════════════════════════════════════════════════════════════════════

with tab_monitor:
    st.markdown("## System monitor")
    ref_int = st.sidebar.slider("System refresh (s)", 2, 30, 5, key="sys_ref_int")

    @st.fragment(run_every=ref_int)
    def _system_panel():
        snap = get_snapshot(disk_paths=["/", "/home", "/media"])

        st.markdown("### CPU")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Usage", f"{snap.cpu.usage_pct:.1f}%")
        sc2.metric("Logical cores", snap.cpu.count_logical)
        sc3.metric("Physical cores", snap.cpu.count_physical)
        sc4.metric("Frequency", f"{snap.cpu.freq_mhz:.0f} MHz" if snap.cpu.freq_mhz else "—")
        st.progress(snap.cpu.usage_pct / 100)

        st.markdown("### RAM")
        sr1, sr2, sr3, sr4 = st.columns(4)
        sr1.metric("Used", f"{snap.ram.used_gb:.1f} GB")
        sr2.metric("Total", f"{snap.ram.total_gb:.1f} GB")
        sr3.metric("Available", f"{snap.ram.available_gb:.1f} GB")
        sr4.metric("Usage %", f"{snap.ram.percent:.1f}%")
        st.progress(snap.ram.percent / 100)
        if snap.ram.swap_total_gb > 0:
            st.caption(f"Swap: {snap.ram.swap_used_gb:.1f} / {snap.ram.swap_total_gb:.1f} GB")

        st.markdown("### GPU")
        if snap.gpus:
            for gpu in snap.gpus:
                mem_pct = gpu.mem_used_mb / gpu.mem_total_mb * 100 if gpu.mem_total_mb else 0
                g1, g2, g3, g4, g5 = st.columns(5)
                g1.metric(f"GPU {gpu.index}", gpu.name[:28])
                g2.metric("VRAM used", f"{gpu.mem_used_mb / 1024:.1f} GB")
                g3.metric("VRAM total", f"{gpu.mem_total_mb / 1024:.1f} GB")
                g4.metric("Utilization", f"{gpu.util_pct}%")
                g5.metric("Temperature", f"{gpu.temp_c}°C")
                st.progress(mem_pct / 100,
                            text=f"VRAM {gpu.mem_used_mb}/{gpu.mem_total_mb} MB ({mem_pct:.1f}%)")
                if gpu.power_w is not None:
                    limit_str = f" / {gpu.power_limit_w:.0f} W" if gpu.power_limit_w else ""
                    st.caption(f"Power draw: {gpu.power_w:.1f} W{limit_str}")
        else:
            st.info("No GPU detected (nvidia-smi unavailable).")

        st.markdown("### Disk")
        if snap.disks:
            disk_cols = st.columns(len(snap.disks))
            for col, disk in zip(disk_cols, snap.disks):
                col.metric(disk.path, f"{disk.free_gb:.1f} GB free")
                col.progress(disk.percent / 100,
                             text=f"{disk.used_gb:.1f} / {disk.total_gb:.1f} GB ({disk.percent:.1f}%)")
        else:
            st.info("Could not read disk usage.")

        st.markdown("### Network (cumulative since boot)")
        nn1, nn2 = st.columns(2)
        nn1.metric("Sent", f"{snap.network.bytes_sent_mb / 1024:.2f} GB")
        nn2.metric("Received", f"{snap.network.bytes_recv_mb / 1024:.2f} GB")

    _system_panel()

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dataset:
    st.markdown("## Data explorer — BigEarthNet-S2 v2.0")

    meta_path: Path | None = None
    for candidate in [
        "/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet",
        "/home/bejeque/alu0101317038/datasets/bigearthnet/metadata.parquet",
    ]:
        if Path(candidate).exists():
            meta_path = Path(candidate)
            break

    st.markdown("### Dataset splits")
    split_col, pie_col = st.columns([1, 1])
    with split_col:
        ds1, ds2 = st.columns(2)
        ds1.metric("Train", f"{SPLIT_SIZES['train']:,}")
        ds2.metric("Validation", f"{SPLIT_SIZES['val']:,}")
        ds3, ds4 = st.columns(2)
        ds3.metric("Test", f"{SPLIT_SIZES['test']:,}")
        ds4.metric("Total patches", f"{sum(SPLIT_SIZES.values()):,}")
    with pie_col:
        fig_splits = go.Figure(go.Pie(
            labels=["Train", "Validation", "Test"],
            values=list(SPLIT_SIZES.values()),
            hole=0.45,
            marker_colors=[COLORS[0], COLORS[2], COLORS[1]],
            textinfo="label+percent",
        ))
        fig_splits.update_layout(
            **_base_layout(260, "Split distribution", margin=dict(l=10, r=10, t=40, b=10)),
            showlegend=False,
        )
        _show(fig_splits, "splits")

    # ── Class distribution ──────────────────────────────────────────────────────
    st.markdown("### Class distribution (train split)")
    dist_df = None
    if meta_path:
        dist_df = _load_class_distribution(str(meta_path))
        if dist_df is not None:
            st.caption(f"Source: {meta_path.name} (real multi-label count)")
        else:
            st.caption("Could not read the parquet — using approximate statistics.")
    if dist_df is None:
        dist_df = class_distribution_approximate()
        if not meta_path:
            st.caption("metadata.parquet not found — using approximate statistics.")

    dist_df = dist_df.sort_values("train_count", ascending=True).reset_index(drop=True)
    dist_df["color"] = dist_df["train_count"].apply(
        lambda v: COLORS[3] if v < 10000 else (COLORS[1] if v < 40000 else COLORS[2])
    )
    fig_dist = go.Figure(go.Bar(
        y=dist_df["class"], x=dist_df["train_count"],
        orientation="h",
        marker_color=dist_df["color"].tolist(),
        text=dist_df["train_count"].apply(lambda v: f"{v:,}").tolist(),
        textposition="outside",
        cliponaxis=False,
    ))
    fig_dist.update_layout(
        **_base_layout(620, "Samples per class (train)", margin=dict(l=300, r=90, t=40, b=40)),
        xaxis_title="Number of samples (multi-label)", yaxis_title="",
    )
    fig_dist.update_yaxes(tickfont=dict(size=11), automargin=True)
    max_x = dist_df["train_count"].max()
    fig_dist.update_xaxes(range=[0, max_x * 1.15])
    _show(fig_dist, "class_distribution")
    st.caption(
        "Red = rare class (<10K), Orange = moderate (<40K), Green = frequent. "
        "Being multi-label, the sum of labels exceeds the number of patches. "
        "Rare classes cap the macro-F1 ceiling."
    )

    st.markdown("### Class imbalance")
    max_c = dist_df["train_count"].max()
    min_c = dist_df["train_count"].min()
    ratio = max_c / min_c if min_c > 0 else float("inf")
    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Most frequent class", dist_df.iloc[-1]["class"][:28],
               f"{int(max_c):,}")
    ci2.metric("Rarest class", dist_df.iloc[0]["class"][:28],
               f"{int(min_c):,}")
    ci3.metric("Imbalance ratio", f"{ratio:.1f}×")
    _dl_csv(dist_df[["class", "train_count"]], "class_distribution.csv",
            "Download distribution")

    # ── Example images per class ────────────────────────────────────────────────
    st.markdown("### Example images per class")
    if not meta_path:
        st.info("Requires dataset access (metadata.parquet not found).")
    else:
        # Path to the dataset root directory (next to the parquet)
        ds_root = meta_path.parent / "BigEarthNet-S2"
        if not ds_root.exists():
            st.info(f"Dataset directory not found at {ds_root}.")
        else:
            sel_class = st.selectbox(
                "Class", CLASS_NAMES,
                index=CLASS_NAMES.index("Marine waters") if "Marine waters" in CLASS_NAMES else 0,
            )
            with st.spinner("Loading RGB images from the dataset…"):
                examples = _load_example_images(str(meta_path), str(ds_root), sel_class, n=4)
            if examples:
                st.caption(
                    "RGB proxy (Sentinel-2 bands B04/B03/B02) with percentile stretch "
                    "for visibility. Each patch is 120×120 px (~1.2 km²)."
                )
                img_cols = st.columns(len(examples))
                for col, (pid, img) in zip(img_cols, examples):
                    col.image(img, caption=pid.split("_")[-2] + "_" + pid.split("_")[-1],
                              use_container_width=True)
            else:
                st.warning(
                    f"Could not load images for '{sel_class}'. "
                    "Is the dataset complete and accessible?"
                )

    # ── Country ──────────────────────────────────────────────────────────────────
    if meta_path:
        country_counts = get_country_distribution(meta_path)
        if country_counts is not None and not country_counts.empty:
            st.markdown("### Distribution by country (train)")
            top_n = country_counts.head(15).sort_values(ascending=True)
            fig_c = go.Figure(go.Bar(
                x=top_n.values, y=top_n.index, orientation="h",
                marker_color=COLORS[0], opacity=0.85,
                text=[f"{v:,}" for v in top_n.values], textposition="outside",
                cliponaxis=False,
            ))
            fig_c.update_layout(
                **_base_layout(420, "Top 15 countries by number of patches",
                               margin=dict(l=120, r=80, t=40, b=40)),
                xaxis_title="Patches", yaxis_title="",
            )
            fig_c.update_xaxes(range=[0, top_n.values.max() * 1.15])
            _show(fig_c, "countries")

    # ── Difficulty vs frequency ──────────────────────────────────────────────────
    st.markdown("### Per-class difficulty vs frequency")
    st.caption("Crosses each class's frequency with its validation F1 (most recent per-class CSV).")
    perclass_csvs_all = sorted(ROOT.rglob("perclass_metrics_*.csv"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
    if perclass_csvs_all:
        pc_df = parse_perclass_csv(perclass_csvs_all[0])
        if not pc_df.empty:
            last_ep = pc_df["epoch"].max()
            ep_pc = pc_df[pc_df["epoch"] == last_ep].copy()
            ep_pc = ep_pc.merge(dist_df[["class", "train_count"]],
                                left_on="class_name", right_on="class", how="left")
            fig_sc = px.scatter(
                ep_pc, x="train_count", y="f1",
                text="class_name", color="f1",
                color_continuous_scale="RdYlGn", range_color=[0, 1],
                labels={"train_count": "Training samples", "f1": "Val F1"},
                title=f"F1 vs class frequency (epoch {last_ep})",
            )
            fig_sc.update_traces(textposition="top center", textfont_size=9)
            fig_sc.update_layout(
                **_base_layout(480, margin=dict(l=60, r=40, t=40, b=50)),
                showlegend=False,
            )
            fig_sc.update_yaxes(range=[-0.05, 1.05])
            _show(fig_sc, "f1_vs_frequency")
            st.caption(
                "Classes with few samples tend to have low F1. "
                "The points at the bottom-left are the hardest to improve."
            )
    else:
        st.info("No per-class CSV found. Run a training with `--layers confusion`.")

# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_modelos:
    st.markdown("## Model explorer")
    st.caption("Explore timm models and compare parameters and VRAM requirements.")

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

    extra_model = st.text_input("Add a custom timm model", placeholder="e.g. convnext_large")
    if extra_model.strip():
        candidate_models.append(extra_model.strip())

    if not candidate_models:
        st.info("Select at least one family.")
    else:
        with st.spinner("Loading model statistics…"):
            rows = compare_models(candidate_models, [cmp_batch], num_classes=19)

        if not rows:
            st.warning("Could not load any model.")
        else:
            cmp_df = pd.DataFrame(rows)
            vram_col = f"VRAM est. bs={cmp_batch} (GB)"

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
            _dl_csv(cmp_df, "model_comparison.csv", "Download model comparison")

            st.markdown("### Parameters vs FLOPs")
            plot_df = cmp_df[cmp_df["FLOPs (MFLOPs)"] != "—"].copy()
            if not plot_df.empty:
                plot_df["FLOPs (MFLOPs)"] = pd.to_numeric(plot_df["FLOPs (MFLOPs)"], errors="coerce")
                plot_df[vram_col] = pd.to_numeric(plot_df.get(vram_col, 0), errors="coerce").fillna(1)
                fig_bubble = px.scatter(
                    plot_df, x="FLOPs (MFLOPs)", y="Params (M)",
                    size=vram_col, color="Family", text="Model", hover_name="Model",
                    size_max=40,
                    labels={"FLOPs (MFLOPs)": "FLOPs per image (MFLOPs)", "Params (M)": "Parameters (M)"},
                )
                fig_bubble.update_traces(textposition="top center", textfont_size=8)
                fig_bubble.update_layout(**_base_layout(420, "Model complexity"), showlegend=True)
                _show(fig_bubble, "model_complexity")
                st.caption("Bubble size = estimated VRAM at the selected batch size.")

            st.markdown("### Required VRAM by batch size")
            vram_models = candidate_models[:8]
            with st.spinner("Computing VRAM estimates…"):
                vram_rows = compare_models(vram_models, [4, 8, 16, 32, 64, 128])

            if vram_rows:
                vram_df = pd.DataFrame(vram_rows)
                vram_cols_list = [c for c in vram_df.columns if c.startswith("VRAM")]
                fig_vram_m = go.Figure()
                for col_v in vram_cols_list:
                    bs_val = col_v.split("bs=")[1].split(" ")[0]
                    fig_vram_m.add_trace(go.Bar(
                        name=f"bs={bs_val}", x=vram_df["Model"],
                        y=pd.to_numeric(vram_df[col_v], errors="coerce"),
                    ))
                fig_vram_m.add_hline(y=8, line_dash="dash", line_color="red",
                                     annotation_text="RTX 3060 Ti (8 GB)")
                fig_vram_m.add_hline(y=32, line_dash="dash", line_color="orange",
                                     annotation_text="V100 (32 GB)")
                fig_vram_m.update_layout(
                    **_base_layout(380, "Estimated VRAM (GB) by batch size"),
                    barmode="group", xaxis_tickangle=30, yaxis_title="GB",
                )
                _show(fig_vram_m, "vram_by_batch")

            st.markdown("### Quick launch")
            selected_for_launch = st.selectbox(
                "Select a model to prefill the Launcher", [r["Model"] for r in rows]
            )
            if selected_for_launch:
                st.info(f"Go to the **Launcher** tab and use the model `{selected_for_launch}`.")
                st.session_state["preselected_model"] = selected_for_launch

# ═══════════════════════════════════════════════════════════════════════════════
# DDP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ddp:
    st.markdown("## DDP analysis — Single-GPU vs Distributed")
    st.caption("Compares single-GPU and DDP runs of the same model to measure real speedup, efficiency and scalability.")

    all_runs_ddp = _get_runs()
    if not all_runs_ddp:
        st.info("No runs found.")
    else:
        single_runs = [r for r in all_runs_ddp if r.mode == "single"]
        # Any distributed mode counts as DDP: "ddp" (multi-process NCCL) and
        # "ddp_hetero" (GPU+CPU). Previously only the exact "ddp" was filtered, so
        # heterogeneous runs did not appear in this tab.
        ddp_runs = [r for r in all_runs_ddp if r.mode.startswith("ddp")]

        da1, da2, da3 = st.columns(3)
        da1.metric("Single-GPU runs", len(single_runs))
        da2.metric("Distributed runs", len(ddp_runs))
        da3.metric("Total runs", len(all_runs_ddp))

        if not ddp_runs:
            st.info("No DDP runs yet. Launch `scripts/train_ddp.py` — results will appear here automatically.")
        else:
            st.markdown("### DDP runs")
            ddp_rows = []
            for r in ddp_runs:
                try:
                    ddf = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
                    if ddf.empty:
                        continue
                    best_f1 = _safe_max(ddf["val_f1"]) if "val_f1" in ddf.columns else float("nan")
                    avg_ep = (ddf["epoch_time"].dropna().mean()
                              if "epoch_time" in ddf.columns and ddf["epoch_time"].notna().any() else None)
                    ddp_rows.append({
                        "Run": r.label[:50], "Model": r.model or "—",
                        "Mode": r.mode, "Environment": r.env,
                        "Best Val F1": round(best_f1, 4),
                        "Epochs": len(ddf),
                        "Avg epoch (min)": round(avg_ep / 60, 1) if avg_ep else "—",
                    })
                except Exception:
                    pass
            if ddp_rows:
                st.dataframe(pd.DataFrame(ddp_rows), use_container_width=True, hide_index=True)

        if single_runs and ddp_runs:
            st.markdown("### Speedup analysis")
            col_s, col_d = st.columns(2)
            with col_s:
                single_lbl = st.selectbox("Single-GPU run", [r.label for r in single_runs], key="ddp_single_sel")
            with col_d:
                ddp_lbl = st.selectbox("DDP run", [r.label for r in ddp_runs], key="ddp_ddp_sel")

            r_single = next(r for r in single_runs if r.label == single_lbl)
            r_ddp = next(r for r in ddp_runs if r.label == ddp_lbl)

            df_s = _load_df(str(r_single.log_path), str(r_single.epoch_csv_path) if r_single.epoch_csv_path else None)
            df_d = _load_df(str(r_ddp.log_path), str(r_ddp.epoch_csv_path) if r_ddp.epoch_csv_path else None)

            if not df_s.empty and not df_d.empty:
                avg_s = df_s["epoch_time"].mean() if "epoch_time" in df_s.columns and df_s["epoch_time"].notna().any() else None
                avg_d = df_d["epoch_time"].mean() if "epoch_time" in df_d.columns and df_d["epoch_time"].notna().any() else None

                # Correct labels by distribution type: the heterogeneous case
                # is V100+CPU (NOT 2 equivalent GPUs).
                is_hetero = r_ddp.mode == "ddp_hetero"
                worker_desc = "V100 + CPU" if is_hetero else "2 GPUs"
                n_workers = 2

                su1, su2, su3, su4 = st.columns(4)
                su1.metric("Single-GPU epoch", f"{avg_s/60:.1f} min" if avg_s else "—")
                su2.metric(f"Distributed epoch ({worker_desc})", f"{avg_d/60:.1f} min" if avg_d else "—")
                speedup = None
                if avg_s and avg_d and avg_d > 0:
                    speedup = avg_s / avg_d
                    su3.metric("Real speedup", f"{speedup:.2f}×")
                    su4.metric(f"Efficiency vs ideal {n_workers}× ", f"{speedup / n_workers * 100:.1f}%")

                if speedup is not None and speedup < 1:
                    st.warning(
                        f"**Speedup < 1×: distributed is {1/speedup:.1f}× SLOWER** than the GPU alone. "
                        + ("This is the expected result of **synchronous** DDP with imbalanced hardware "
                           "(V100 + CPU): on every batch the GPU waits for the CPU (~50× slower), "
                           "so the system runs at the pace of the slowest node. It shows *when NOT to distribute*."
                           if is_hetero else
                           "Check the load balancing / inter-GPU communication.")
                    )
                elif speedup is not None:
                    st.success(
                        f"**Speedup {speedup:.2f}× with {worker_desc}** "
                        f"(efficiency {speedup/n_workers*100:.0f}% over the ideal linear {n_workers}×)."
                    )

                fig_ddp_f1 = go.Figure()
                if "val_f1" in df_s.columns:
                    fig_ddp_f1.add_trace(go.Scatter(x=df_s["epoch"], y=df_s["val_f1"],
                                                     name="Single-GPU Val F1", line=dict(color=COLORS[0], width=2)))
                if "val_f1" in df_d.columns:
                    fig_ddp_f1.add_trace(go.Scatter(x=df_d["epoch"], y=df_d["val_f1"],
                                                     name="DDP Val F1", line=dict(color=COLORS[2], width=2)))
                fig_ddp_f1.update_layout(**_base_layout(300, "Val F1: Single-GPU vs DDP"),
                                          xaxis_title="Epoch", yaxis_title="Val F1")
                _show(fig_ddp_f1, "ddp_f1")

                if avg_s and avg_d:
                    fig_time_ddp = go.Figure()
                    if "epoch_time" in df_s.columns:
                        fig_time_ddp.add_trace(go.Scatter(x=df_s["epoch"], y=df_s["epoch_time"] / 60,
                                                           name="Single-GPU", line=dict(color=COLORS[0], width=2)))
                    if "epoch_time" in df_d.columns:
                        fig_time_ddp.add_trace(go.Scatter(x=df_d["epoch"], y=df_d["epoch_time"] / 60,
                                                           name="DDP", line=dict(color=COLORS[2], width=2)))
                    fig_time_ddp.update_layout(**_base_layout(260, "Epoch time: Single-GPU vs DDP"),
                                               xaxis_title="Epoch", yaxis_title="Minutes")
                    _show(fig_time_ddp, "ddp_time")

                st.markdown("### Theoretical vs real scaling")
                world_sizes = [1, 2, 4, 8]
                if avg_s:
                    theoretical = [avg_s / ws for ws in world_sizes]
                    fig_scale = go.Figure()
                    fig_scale.add_trace(go.Scatter(
                        x=world_sizes, y=[t / 60 for t in theoretical],
                        name="Theoretical (100% efficiency)",
                        line=dict(color=COLORS[4], width=2, dash="dash"), mode="lines+markers",
                    ))
                    if avg_d:
                        fig_scale.add_trace(go.Scatter(
                            x=[2], y=[avg_d / 60], name=f"Real ({worker_desc})",
                            mode="markers", marker=dict(color=COLORS[2], size=14, symbol="star"),
                        ))
                    fig_scale.update_layout(**_base_layout(300, "Epoch time vs number of workers"),
                                            xaxis_title="Number of workers (processes)", yaxis_title="Minutes per epoch")
                    fig_scale.update_xaxes(tickvals=world_sizes)
                    _show(fig_scale, "ddp_scaling")
                    st.caption(
                        "The theoretical line assumes adding workers IDENTICAL to the single-GPU "
                        "(perfect linear scaling). The real point falls below it due to "
                        "communication overhead, the NFS bottleneck and — in the heterogeneous case — "
                        "because the second worker is a CPU ~50× slower, not another V100."
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# CURVES
# ═══════════════════════════════════════════════════════════════════════════════

with tab_curvas:
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
        df = _load_df(
            str(run.log_path),
            str(run.epoch_csv_path) if run.epoch_csv_path else None,
        )

        if df.empty:
            st.error("Could not parse any epoch from the selected run.")
        else:
            n_epochs = len(df)
            best_f1 = _safe_max(df["val_f1"]) if "val_f1" in df.columns else float("nan")
            best_ep_v = _safe_val_at_best(df, "val_f1", "epoch")
            best_epoch = int(best_ep_v) if best_ep_v is not None else "—"
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
                c5.metric("Total duration", _dur_str(df["epoch_time"].sum()))

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
                _show(_metric_fig(df, "train_f1", "val_f1", "F1 (macro)", "F1", extra_traces=extra_thresh), "f1")
                _show(_metric_fig(df, "train_loss", "val_loss", "Loss (BCE)", "Loss"), "loss")
            with c2:
                _show(_metric_fig(df, "train_acc", "val_acc", "Accuracy", "Accuracy",
                                  color_train=COLORS[4], color_val=COLORS[5]), "accuracy")
                _show(_metric_fig(df, "val_prec", "val_rec", "Precision & Recall (val)",
                                  "Score", color_train=COLORS[2], color_val=COLORS[3]), "prec_rec")

            if "epoch_time" in df.columns and df["epoch_time"].notna().any():
                et = df[["epoch", "epoch_time"]].dropna()
                fig_et = go.Figure(go.Bar(x=et["epoch"], y=et["epoch_time"] / 60,
                                          marker_color=COLORS[0], opacity=0.8))
                fig_et.update_layout(**_base_layout(240, "Time per epoch (min)"),
                                     xaxis_title="Epoch", yaxis_title="Minutes")
                _show(fig_et, "epoch_time")

            has_energy = "energy_eval_wh" in df.columns and df["energy_eval_wh"].notna().any()
            if has_energy:
                st.markdown("#### Energy consumption")
                e1, e2 = st.columns(2)
                with e1:
                    rows_e = []
                    for _, row in df.iterrows():
                        if pd.notna(row.get("energy_eval_wh")):
                            entry = {"epoch": row["epoch"], "Eval (Wh)": row["energy_eval_wh"]}
                            if pd.notna(row.get("energy_train_j")):
                                entry["Train (Wh)"] = row["energy_train_j"] / 3600
                            rows_e.append(entry)
                    if rows_e:
                        df_e = pd.DataFrame(rows_e)
                        fig_e = go.Figure()
                        if "Train (Wh)" in df_e.columns:
                            fig_e.add_trace(go.Bar(x=df_e["epoch"], y=df_e["Train (Wh)"],
                                                   name="Train", marker_color=COLORS[0], opacity=0.85))
                        fig_e.add_trace(go.Bar(x=df_e["epoch"], y=df_e["Eval (Wh)"],
                                               name="Eval", marker_color=COLORS[1], opacity=0.85))
                        fig_e.update_layout(**_base_layout(260, "Energy per epoch (Wh)"),
                                            barmode="group", xaxis_title="Epoch", yaxis_title="Wh")
                        _show(fig_e, "energy")
                with e2:
                    power_cols = []
                    if "power_eval_w" in df.columns and df["power_eval_w"].notna().any():
                        power_cols.append(("Eval power (W)", "power_eval_w", COLORS[1]))
                    if "power_train_w" in df.columns and df["power_train_w"].notna().any():
                        power_cols.append(("Train power (W)", "power_train_w", COLORS[0]))
                    if power_cols:
                        fig_p = go.Figure()
                        for name_p, col_p, color_p in power_cols:
                            fig_p.add_trace(go.Scatter(x=df["epoch"], y=df[col_p],
                                                       name=name_p, mode="lines+markers",
                                                       line=dict(color=color_p, width=2)))
                        fig_p.update_layout(**_base_layout(260, "Average GPU power per epoch (W)"),
                                            xaxis_title="Epoch", yaxis_title="Watts")
                        _show(fig_p, "gpu_power")

                total_eval_wh = df["energy_eval_wh"].sum()
                total_train_wh = (df["energy_train_j"].sum() / 3600
                                  if "energy_train_j" in df.columns and df["energy_train_j"].notna().any() else 0)
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("Total eval energy", f"{total_eval_wh:.1f} Wh")
                if total_train_wh > 0:
                    ec2.metric("Total train energy", f"{total_train_wh:.1f} Wh")
                    ec3.metric("Total energy", f"{total_eval_wh + total_train_wh:.1f} Wh")

            _dl_csv(df, "epoch_metrics.csv", "Download epoch_metrics.csv")

            with st.expander("Full epochs table"):
                st.dataframe(df.set_index("epoch"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PER-CLASS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_porclase:
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
        subtab_bars, subtab_trend, subtab_cm = st.tabs(
            ["Per-class", "Trend", "Confusion matrix"]
        )

        with subtab_bars:
            if run.perclass_csv_path and run.perclass_csv_path.exists():
                pcdf = _load_perclass(str(run.perclass_csv_path))
                epochs_available = sorted(pcdf["epoch"].unique().tolist())
                selected_ep = st.selectbox("Epoch", epochs_available, format_func=lambda e: f"Epoch {e}")
                ep_df = pcdf[pcdf["epoch"] == selected_ep].copy().sort_values("f1", ascending=False)

                styled = (
                    ep_df[["class_name", "f1", "precision", "recall"]]
                    .style.map(_color_f1_cell, subset=["f1"])
                    .format({"f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"})
                )
                st.dataframe(styled, use_container_width=True, height=280)
                _dl_csv(ep_df[["class_name", "f1", "precision", "recall"]],
                        f"perclass_ep{selected_ep}.csv", "Download per-class table")

                colors_f1 = [
                    COLORS[2] if v >= 0.6 else (COLORS[1] if v >= 0.3 else COLORS[3])
                    for v in ep_df["f1"]
                ]
                fig_pc = go.Figure()
                fig_pc.add_trace(go.Bar(y=ep_df["class_name"], x=ep_df["precision"],
                                        name="Precision", orientation="h",
                                        marker_color=COLORS[0], opacity=0.8))
                fig_pc.add_trace(go.Bar(y=ep_df["class_name"], x=ep_df["recall"],
                                        name="Recall", orientation="h",
                                        marker_color=COLORS[1], opacity=0.8))
                fig_pc.add_trace(go.Bar(y=ep_df["class_name"], x=ep_df["f1"],
                                        name="F1", orientation="h", marker_color=colors_f1))
                fig_pc.update_layout(
                    barmode="group",
                    title=dict(text=f"Per-class metrics — Epoch {selected_ep}", font=dict(size=13)),
                    xaxis_title="Score", xaxis=dict(range=[0, 1]),
                    height=600, margin=dict(l=200, r=16, t=36, b=40),
                    paper_bgcolor="white", plot_bgcolor="#f8fafc",
                )
                _show(fig_pc, f"per_class_ep{selected_ep}")
            else:
                st.info("No per-class data. Use `--layers confusion` to generate it.")

        with subtab_trend:
            if run.perclass_csv_path and run.perclass_csv_path.exists():
                pcdf = _load_perclass(str(run.perclass_csv_path))
                classes = sorted(pcdf["class_name"].unique().tolist())
                col_sel, col_met = st.columns([3, 1])
                with col_sel:
                    selected_classes = st.multiselect("Classes (max 8)", classes, default=classes[:4], max_selections=8)
                with col_met:
                    metric_sel = st.radio("Metric", ["f1", "precision", "recall"])

                if selected_classes:
                    fig_trend = go.Figure()
                    for i, cls in enumerate(selected_classes):
                        cdf = pcdf[pcdf["class_name"] == cls].sort_values("epoch")
                        fig_trend.add_trace(go.Scatter(
                            x=cdf["epoch"], y=cdf[metric_sel],
                            name=cls[:30], mode="lines+markers",
                            line=dict(color=COLORS[i % len(COLORS)], width=2), marker=dict(size=4),
                        ))
                    fig_trend.update_layout(
                        **_base_layout(400, f"{metric_sel.capitalize()} per class across epochs"),
                        xaxis_title="Epoch",
                    )
                    fig_trend.update_yaxes(range=[0, 1])
                    _show(fig_trend, "class_trend")
            else:
                st.info("No per-class CSV for this run.")

        with subtab_cm:
            if run.confusion_matrix_csv_path and run.confusion_matrix_csv_path.exists():
                cm_df = parse_confusion_matrix_csv(run.confusion_matrix_csv_path)
                epochs_cm = sorted(cm_df["epoch"].unique().tolist())

                col_cm1, col_cm2 = st.columns([3, 1])
                with col_cm1:
                    selected_cm_ep = st.selectbox("Epoch", epochs_cm,
                                                   format_func=lambda e: f"Epoch {e}", key="cm_epoch_sel")
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

                shapes = []
                for group_name, (idxs, color) in _CLASS_GROUPS.items():
                    positions = [i for i, cls in enumerate(class_order) if i in idxs]
                    if not positions:
                        continue
                    lo, hi = min(positions), max(positions)
                    shapes.append(dict(
                        type="rect", x0=lo - 0.5, x1=hi + 0.5, y0=lo - 0.5, y1=hi + 0.5,
                        line=dict(color=color, width=2.5), fillcolor="rgba(0,0,0,0)", layer="above",
                    ))

                fig_cm = go.Figure(go.Heatmap(
                    z=z_plot, x=class_order, y=class_order,
                    colorscale="Blues", zmin=zmin, zmax=zmax,
                    text=text, texttemplate="%{text}", textfont={"size": 8},
                    hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z:.3f}<extra></extra>",
                    colorbar=dict(title=cb_title),
                ))
                fig_cm.update_layout(
                    title=dict(text=f"Confusion matrix ({cm_mode.lower()}) — Epoch {selected_cm_ep}",
                               font=dict(size=13)),
                    xaxis=dict(title="Predicted", tickangle=45, tickfont=dict(size=9),
                               tickmode="array", tickvals=list(range(n_classes)), ticktext=class_order),
                    yaxis=dict(title="True", tickfont=dict(size=9), autorange="reversed",
                               tickmode="array", tickvals=list(range(n_classes)), ticktext=class_order),
                    height=660, margin=dict(l=180, r=20, t=50, b=180),
                    paper_bgcolor="white", shapes=shapes,
                )
                _show(fig_cm, f"confusion_matrix_ep{selected_cm_ep}")

                legend_html = " &nbsp; ".join(
                    f'<span style="display:inline-block;width:12px;height:12px;background:{color};'
                    f'border-radius:2px;margin-right:4px;vertical-align:middle"></span>'
                    f'<span style="font-size:0.8rem">{name}</span>'
                    for name, (_, color) in _CLASS_GROUPS.items()
                )
                st.markdown(
                    f"<div style='margin-top:4px'>{legend_html}</div>"
                    "<div style='font-size:0.75rem;color:#64748b;margin-top:4px'>"
                    "The colored borders group classes by ecosystem type. "
                    "The diagonal = per-class recall.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No confusion matrix. Use `--layers confusion` to generate it.")

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH
# ═══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    elif not run.batch_csv_path:
        st.info(
            "No batch-level CSV for this run. "
            "Use `--layers batch-monitor` to generate it. "
            "With `--batch-log-every 1` you get one record per individual batch."
        )
    else:
        # ── Sub-tabs: per epoch | global history | learning rate ───────────────
        subtab_by_ep, subtab_global, subtab_lr = st.tabs(
            ["Per epoch", "Global history", "Learning rate"]
        )

        # Load with a short TTL so the live refresh works
        @st.cache_data(ttl=5)
        def _load_batch_live(p: str) -> pd.DataFrame:
            return _load_batch(p)

        bdf = _load_batch_live(str(run.batch_csv_path))
        has_batch_loss = "batch_loss" in bdf.columns and bdf["batch_loss"].notna().any()
        has_lr = "lr" in bdf.columns and bdf["lr"].notna().any()

        # Map of available per-batch metrics → readable label
        _BATCH_METRIC_LABELS = {
            "running_loss": "Running mean loss",
            "batch_loss": "Instantaneous batch loss",
            "batch_f1": "F1 (macro) per batch",
            "batch_acc": "Accuracy per batch",
            "batch_prec": "Precision (macro) per batch",
        }

        def _available_batch_metrics() -> list[str]:
            opts = ["running_loss"]
            for col in ("batch_loss", "batch_f1", "batch_acc", "batch_prec"):
                if col in bdf.columns and bdf[col].notna().any():
                    opts.append(col)
            return opts

        def _is_loss_metric(m: str) -> bool:
            return "loss" in m

        if bdf.empty:
            st.warning("The batch CSV is empty.")
        else:
            epochs_available_b = sorted(bdf["epoch"].unique())
            n_batches_total = int(bdf["n_batches"].iloc[0]) if not bdf.empty else "—"

            # Quick summary
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Epochs recorded", len(epochs_available_b))
            bc2.metric("Batches per epoch", n_batches_total)
            bc3.metric("Total records", len(bdf))
            if has_lr:
                bc4.metric("Initial LR", f"{bdf['lr'].iloc[0]:.2e}")

            # ── Tab: per epoch ─────────────────────────────────────────────────
            with subtab_by_ep:
                col_ep, col_met, col_ma = st.columns([2, 2, 2])
                with col_ep:
                    selected_epochs_b = st.multiselect(
                        "Epochs", epochs_available_b,
                        default=list(epochs_available_b[-min(3, len(epochs_available_b)):]),
                    )
                with col_met:
                    batch_metric = st.selectbox(
                        "Metric", _available_batch_metrics(),
                        format_func=lambda m: _BATCH_METRIC_LABELS.get(m, m),
                    )
                with col_ma:
                    ma_window = st.slider("Moving average (batches)", 0, 200, 10,
                                          help="0 = disabled")

                if selected_epochs_b and batch_metric in bdf.columns:
                    fig_b = go.Figure()
                    for i, ep in enumerate(selected_epochs_b):
                        subset = bdf[bdf["epoch"] == ep].copy().sort_values("batch")
                        color = COLORS[i % len(COLORS)]

                        # Base line (semi-transparent)
                        fig_b.add_trace(go.Scatter(
                            x=subset["batch"], y=subset[batch_metric],
                            name=f"Ep {ep}", mode="lines",
                            line=dict(color=color, width=1), opacity=0.35,
                            legendgroup=f"ep{ep}",
                        ))

                        if ma_window > 1 and len(subset) >= ma_window:
                            ma = subset[batch_metric].rolling(ma_window, center=True).mean()
                            fig_b.add_trace(go.Scatter(
                                x=subset["batch"], y=ma,
                                name=f"Ep {ep} MA{ma_window}", mode="lines",
                                line=dict(color=color, width=2.5),
                                legendgroup=f"ep{ep}",
                            ))

                        # Spike detection
                        mean_l = subset[batch_metric].mean()
                        std_l = subset[batch_metric].std()
                        if not pd.isna(std_l) and std_l > 0:
                            spikes = subset[subset[batch_metric] > mean_l + 2.5 * std_l]
                            if not spikes.empty:
                                fig_b.add_trace(go.Scatter(
                                    x=spikes["batch"], y=spikes[batch_metric],
                                    name=f"Spike Ep{ep}", mode="markers",
                                    marker=dict(color="red", size=7, symbol="x"),
                                    legendgroup=f"ep{ep}", showlegend=False,
                                ))

                    y_label = _BATCH_METRIC_LABELS.get(batch_metric, batch_metric)
                    fig_b.update_layout(
                        **_base_layout(420, f"{y_label}"),
                        xaxis_title="Batch within epoch",
                        yaxis_title=y_label,
                    )
                    # F1/acc/prec metrics live in [0,1]
                    if not _is_loss_metric(batch_metric):
                        fig_b.update_yaxes(range=[0, 1])
                    _show(fig_b, f"batch_{batch_metric}_per_epoch")
                    sel_bdf = bdf[bdf["epoch"].isin(selected_epochs_b)]
                    _dl_csv(sel_bdf, "batch_metrics_sel.csv", "Download selected data")

                    with st.expander("Raw data"):
                        st.dataframe(sel_bdf, use_container_width=True)

            # ── Tab: global history (x axis = global batch) ────────────────────
            with subtab_global:
                st.markdown(
                    "Full view of the entire training history on a single axis. "
                    "The vertical lines mark the epoch boundaries."
                )
                col_gm, col_gma = st.columns([2, 2])
                with col_gm:
                    global_metric = st.selectbox(
                        "Global metric", _available_batch_metrics(),
                        format_func=lambda m: _BATCH_METRIC_LABELS.get(m, m),
                        key="global_metric_sel",
                    )
                with col_gma:
                    gma_window = st.slider("Moving average (global batches)", 0, 500, 50,
                                           help="0 = disabled", key="global_ma")

                if global_metric in bdf.columns:
                    all_sorted = bdf.sort_values("global_batch")
                    fig_g = go.Figure()

                    # Full series (semi-transparent)
                    fig_g.add_trace(go.Scatter(
                        x=all_sorted["global_batch"], y=all_sorted[global_metric],
                        name="Data", mode="lines",
                        line=dict(color=COLORS[0], width=1), opacity=0.3,
                    ))

                    if gma_window > 1 and len(all_sorted) >= gma_window:
                        gma = all_sorted[global_metric].rolling(gma_window, center=True).mean()
                        fig_g.add_trace(go.Scatter(
                            x=all_sorted["global_batch"], y=gma,
                            name=f"MA{gma_window}", mode="lines",
                            line=dict(color=COLORS[0], width=2.5),
                        ))

                    # Vertical lines per epoch
                    epoch_boundaries = bdf.groupby("epoch")["global_batch"].max()
                    for ep, gb in epoch_boundaries.items():
                        fig_g.add_vline(
                            x=gb, line_dash="dot", line_color="#94a3b8", line_width=1,
                            annotation_text=f"E{ep}", annotation_position="top",
                            annotation_font_size=9,
                        )

                    y_label_g = _BATCH_METRIC_LABELS.get(global_metric, global_metric)
                    fig_g.update_layout(
                        **_base_layout(420, f"{y_label_g} — full history"),
                        xaxis_title="Global batch",
                        yaxis_title=y_label_g,
                    )
                    if not _is_loss_metric(global_metric):
                        fig_g.update_yaxes(range=[0, 1])
                    _show(fig_g, "batch_global_history")
                    _dl_csv(bdf, "batch_metrics_full.csv", "Download full history")

            # ── Tab: learning rate ─────────────────────────────────────────────
            with subtab_lr:
                if not has_lr:
                    st.info(
                        "No learning-rate data. "
                        "Requires `--layers batch-monitor` with the current version of BatchMonitorDecorator."
                    )
                else:
                    lr_sorted = bdf.sort_values("global_batch").dropna(subset=["lr"])
                    fig_lr = go.Figure()
                    fig_lr.add_trace(go.Scatter(
                        x=lr_sorted["global_batch"], y=lr_sorted["lr"],
                        name="Learning rate", mode="lines",
                        line=dict(color=COLORS[2], width=2),
                    ))

                    # Mark epoch boundaries
                    epoch_boundaries_lr = bdf.groupby("epoch")["global_batch"].max()
                    for ep, gb in epoch_boundaries_lr.items():
                        fig_lr.add_vline(
                            x=gb, line_dash="dot", line_color="#94a3b8", line_width=1,
                            annotation_text=f"E{ep}", annotation_position="top",
                            annotation_font_size=9,
                        )

                    fig_lr.update_layout(
                        **_base_layout(380, "Learning-rate evolution"),
                        xaxis_title="Global batch",
                        yaxis_title="Learning rate",
                    )
                    # Log scale if the range is large
                    lr_range = lr_sorted["lr"].max() / (lr_sorted["lr"].min() + 1e-12)
                    if lr_range > 100:
                        fig_lr.update_yaxes(type="log")
                    _show(fig_lr, "learning_rate")

                    # LR stats
                    lr_col1, lr_col2, lr_col3 = st.columns(3)
                    lr_col1.metric("Initial LR", f"{lr_sorted['lr'].iloc[0]:.2e}")
                    lr_col2.metric("Final LR", f"{lr_sorted['lr'].iloc[-1]:.2e}")
                    lr_col3.metric("Minimum LR", f"{lr_sorted['lr'].min():.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# OVERLAY / COMPARE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_comparar:
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
            compare_runs_list = [(lbl, all_run_labels[lbl]) for lbl in selected_compare]
            compare_dfs: list[tuple[str, pd.DataFrame]] = []
            for lbl, r in compare_runs_list:
                cdf = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
                compare_dfs.append((lbl[:30], cdf))

            summary_rows = []
            for lbl, r in compare_runs_list:
                cdf = next(d for ll, d in compare_dfs if ll == lbl[:30])
                best_f1_c = _safe_max(cdf["val_f1"]) if "val_f1" in cdf.columns and not cdf.empty else float("nan")
                best_ep_c_v = _safe_val_at_best(cdf, "val_f1", "epoch")
                _last = cdf["val_f1"].dropna() if "val_f1" in cdf.columns and not cdf.empty else pd.Series(dtype=float)
                total_s_c = cdf["epoch_time"].dropna().sum() if "epoch_time" in cdf.columns else float("nan")
                summary_rows.append({
                    "Run": lbl[:50],
                    "Best Val F1": f"{best_f1_c:.4f}" if not pd.isna(best_f1_c) else "—",
                    "Best epoch": int(best_ep_c_v) if best_ep_c_v is not None else "—",
                    "Final F1": f"{_last.iloc[-1]:.4f}" if not _last.empty else "—",
                    "Epochs": len(cdf),
                    "Duration": _dur_str(total_s_c) if not pd.isna(total_s_c) else "—",
                    "Environment": r.env, "Trace": r.trace_mode,
                })

            sum_df = pd.DataFrame(summary_rows).set_index("Run")
            st.dataframe(sum_df, use_container_width=True)
            _dl_csv(sum_df.reset_index(), "runs_comparison.csv", "Download comparison")
            st.markdown("---")

            st.markdown("#### Metric radar at the best epoch")
            radar_metrics = ["val_f1", "train_f1", "val_acc", "val_prec", "val_rec"]
            radar_fig = go.Figure()
            for i, (lbl, cdf) in enumerate(compare_dfs):
                vals = [
                    float(v) if (v := _safe_val_at_best(cdf, "val_f1", m)) is not None else 0.0
                    for m in radar_metrics
                ]
                vals_closed = vals + [vals[0]]
                radar_fig.add_trace(go.Scatterpolar(
                    r=vals_closed, theta=radar_metrics + [radar_metrics[0]],
                    fill="toself", name=lbl[:30],
                    line=dict(color=COLORS[i % len(COLORS)]), opacity=0.6,
                ))
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=360,
                margin=dict(l=60, r=60, t=40, b=40), paper_bgcolor="white",
                title=dict(text="Metrics at the best Val F1 epoch", font=dict(size=13)),
            )
            _show(radar_fig, "radar_comparison")
            st.markdown("---")

            metrics_to_compare = st.multiselect(
                "Metrics to overlay",
                ["val_f1", "val_loss", "train_f1", "train_loss", "val_prec", "val_rec", "epoch_time"],
                default=["val_f1", "val_loss"],
            )
            cols = st.columns(2)
            for idx, col_name in enumerate(metrics_to_compare):
                fig = _overlay_fig(compare_dfs, col=col_name,
                                   title=col_name.replace("_", " "), y_label=col_name)
                with cols[idx % 2]:
                    _show(fig, f"compare_{col_name}")

# ═══════════════════════════════════════════════════════════════════════════════
# VIABILIDAD
# ═══════════════════════════════════════════════════════════════════════════════

with tab_viabilidad:
    (subtab_report, subtab_predreal, subtab_study, subtab_run_feas) = st.tabs(
        ["Report", "Prediction vs reality", "Real study", "Run analysis"]
    )
    # subtab_ddp_opt and subtab_prediction are no longer their own tabs: they are
    # filled as sections inside "Report" and "Prediction vs reality" respectively
    # (containers created further down, in their parent blocks).

    # Shared load of the selected report
    feasibility_csvs = _get_feasibility_csvs()
    if feasibility_csvs:
        selected_feas_path = st.sidebar.selectbox(
            "Feasibility report", [str(p) for p in feasibility_csvs],
            format_func=_feas_label, key="feas_sidebar_sel",
        )
        meta, bdf_feas = parse_feasibility_csv(Path(selected_feas_path))
    else:
        meta, bdf_feas = {}, pd.DataFrame()

    # ── Prediction vs reality (auto-paired with the run in the sidebar) ─────────
    with subtab_predreal:
        st.markdown("### Feasibility prediction vs what actually happened")
        st.caption(
            "The feasibility runs **on 1 GPU**: from there it (A) estimates the "
            "single-GPU time and (B) **predicts** the speedup when distributing. There "
            "is no '2-GPU' feasibility — multi-GPU is a prediction. Below, both are "
            "contrasted with the real trainings of the same model."
        )
        _feas_csvs_pr = _get_feasibility_csvs()
        if not _feas_csvs_pr:
            st.info("No feasibility reports. Generate one in 'Run analysis'.")
        else:
            # Combos (environment · model) that have feasibility
            _combo_csv = {}
            for _p in _feas_csvs_pr:
                _m, _ = parse_feasibility_csv(_p)
                _env = _p.parent.parent.name if _p.parent.parent else "?"
                _combo_csv.setdefault((_env, _m.get("model_name", "?")), _p)
            _combos = list(_combo_csv.keys())
            _def_i = 0
            if selected_run is not None:
                for _i, (_e, _mo) in enumerate(_combos):
                    if _e == selected_run.env and _mo == selected_run.model:
                        _def_i = _i
                        break
            _combo = st.selectbox("What to compare?", _combos, index=_def_i,
                                  format_func=lambda c: f"{c[0]}  ·  {c[1]}", key="pr_combo")
            _env_pr, _mod_pr = _combo
            _feas_p = _combo_csv[_combo]
            _meta_pr, _feas_df_pr = parse_feasibility_csv(_feas_p)
            _nfs_pr = float(_meta_pr.get("nfs_factor", 1.0) or 1.0)
            st.caption(f"Report used: **{_feas_label(str(_feas_p))}**")

            _all_pr = _get_runs()

            def _find_run(modes):
                return next((r for r in _all_pr if r.env == _env_pr
                             and r.model == _mod_pr and r.mode in modes), None)

            def _ep_mean(r):
                if r is None:
                    return None
                _df = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
                if "epoch_time" in _df.columns and _df["epoch_time"].notna().any():
                    return float(_df["epoch_time"].mean())
                return None

            _single_run = _find_run({"single"})
            _ddp_run = _find_run({"ddp"})
            _hetero_run = _find_run({"ddp_hetero"})

            # ── A) On 1 GPU: estimated vs real time ─────────────────────────────
            st.markdown("#### A · On 1 GPU — estimated vs real time")
            if _single_run is None:
                st.info(f"No **single-GPU** run of {_mod_pr} in {_env_pr}.")
            else:
                _act = _load_df(str(_single_run.log_path),
                                str(_single_run.epoch_csv_path) if _single_run.epoch_csv_path else None)
                _bs_av = (sorted(_feas_df_pr["batch_size"].dropna().astype(int).unique().tolist())
                          if (not _feas_df_pr.empty and "batch_size" in _feas_df_pr.columns) else [])
                _tr_av = (sorted(_feas_df_pr["trace_mode"].unique().tolist())
                          if "trace_mode" in _feas_df_pr.columns else ["simple"])
                if not _bs_av or _act.empty:
                    st.info("Missing data (feasibility benchmark or run times).")
                else:
                    _ca = st.columns([1, 1, 2])
                    _bs = _ca[0].selectbox("Batch", _bs_av, index=len(_bs_av) - 1, key="pr_bs")
                    _tr = _ca[1].selectbox("Trace", _tr_av, key="pr_tr")
                    _cmp = build_comparison(meta=_meta_pr, feas_df=_feas_df_pr, actual_df=_act,
                                            batch_size=int(_bs), trace_mode=_tr, nfs_factor=_nfs_pr)
                    if not _cmp:
                        st.warning(f"No feasibility row for batch={_bs}, trace={_tr}.")
                    else:
                        _rows = {r.metric: r for r in _cmp.rows}
                        _tt = _rows.get("Total time / epoch")
                        _thr = _rows.get("Train throughput")
                        m1, m2, m3 = st.columns(3)
                        if _tt and _tt.estimated is not None and _tt.actual is not None:
                            m1.metric("Estimated time/epoch", f"{_tt.estimated:.2f} min")
                            m2.metric("Real time/epoch", f"{_tt.actual:.2f} min",
                                      delta=f"{_tt.error_pct or 0:+.0f}%", delta_color="off")
                        if _thr and _thr.estimated is not None and _thr.actual is not None:
                            m3.metric("Real throughput", f"{_thr.actual:.0f} img/s",
                                      delta=f"estimated {_thr.estimated:.0f}", delta_color="off")
                        if _tt and _tt.error_pct is not None:
                            _e = _tt.error_pct
                            _io = None
                            try:
                                _io = float(_meta_pr.get("dataset", {}).get("io_bottleneck_ratio"))
                            except (TypeError, ValueError, AttributeError):
                                pass
                            if abs(_e) <= 15:
                                st.success(f"**Accurate prediction** — {_e:+.0f}% error in time/epoch.")
                            elif _e < 0:
                                _x = (f" Likely cause: **I/O-bound** (ratio≈{_io:.1f}) — the synthetic "
                                      "benchmark does not include disk reads (NFS)."
                                      if _io and _io > 1 else "")
                                st.warning(f"**Optimistic estimate** — the real run was {abs(_e):.0f}% slower.{_x}")
                            else:
                                st.info(f"**Pessimistic estimate** — the real run was {_e:.0f}% faster than expected.")
                        # Simple chart: estimated vs real time, same unit (min)
                        _tm = [(n, _rows.get(k)) for n, k in
                               (("Train", "Train time / epoch"),
                                ("Eval", "Eval time / epoch"),
                                ("Total", "Total time / epoch"))]
                        _tm = [(n, r) for n, r in _tm
                               if r and r.estimated is not None and r.actual is not None]
                        if _tm:
                            _names = [n for n, _ in _tm]
                            _fig = go.Figure()
                            _fig.add_trace(go.Bar(name="Estimated", x=_names, y=[r.estimated for _, r in _tm],
                                                  marker_color="#94a3b8",
                                                  text=[f"{r.estimated:.2f}" for _, r in _tm], textposition="outside"))
                            _fig.add_trace(go.Bar(name="Real", x=_names, y=[r.actual for _, r in _tm],
                                                  marker_color="#2563eb",
                                                  text=[f"{r.actual:.2f}" for _, r in _tm], textposition="outside"))
                            _fig.update_layout(**_base_layout(300, "Time per epoch: estimated vs real"),
                                               barmode="group", yaxis_title="Minutes", xaxis_title="")
                            _show(_fig, "pred_time_bars")
                            st.caption("Both bars in each pair at the same height = accurate prediction "
                                       "(grey = estimated, blue = real). Throughput/VRAM/energy in the detail.")
                        with st.expander("See detail and formulas"):
                            _t = _cmp.to_dataframe()
                            st.dataframe(_t, use_container_width=True, hide_index=True)
                            _dl_csv(_t, "prediction_1gpu.csv", "Download")

            # ── B) When distributing: predicted vs real speedup ─────────────────
            st.divider()
            st.markdown("#### B · When distributing — predicted vs real speedup (2 GPUs)")
            _ddp_scen = parse_ddp_scenarios(_meta_pr)
            _pred_sp = None
            if not _ddp_scen.empty and {"n_gpus", "speedup"}.issubset(_ddp_scen.columns):
                _r2 = _ddp_scen[_ddp_scen["n_gpus"] == 2]
                if not _r2.empty:
                    _pred_sp = float(_r2.iloc[0]["speedup"])
            _s_ep = _ep_mean(_single_run)
            _d_ep = _ep_mean(_ddp_run)
            if _single_run is None or _ddp_run is None:
                _msg = ("To measure the **real** speedup you need a single-GPU run **and** a "
                        "multi-GPU DDP run of the same model/environment.")
                if _pred_sp is not None:
                    _msg += f" The feasibility **predicts {_pred_sp:.2f}×** with 2 GPUs."
                if _hetero_run is not None:
                    _msg += (" There is a **heterogeneous** run (V100+CPU), which is not comparable to the "
                             "homogeneous 2-GPU prediction — its speedup is in "
                             "**Comparison → Single vs Distributed**.")
                st.info(_msg)
            elif _s_ep and _d_ep:
                _real_sp = _s_ep / _d_ep
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Predicted speedup", f"{_pred_sp:.2f}×" if _pred_sp else "—")
                sc2.metric("Real speedup", f"{_real_sp:.2f}×")
                if _pred_sp:
                    _serr = (_pred_sp - _real_sp) / _real_sp * 100
                    sc3.metric("Prediction error", f"{_serr:+.0f}%")
                    if abs(_serr) <= 15:
                        st.success(f"**The feasibility predicted the scaling well** — predicted "
                                   f"{_pred_sp:.2f}× vs **{_real_sp:.2f}× real**.")
                    else:
                        st.warning(f"Predicted {_pred_sp:.2f}× vs **{_real_sp:.2f}× real** "
                                   f"(error {_serr:+.0f}%).")
                st.caption(f"Real = single time/epoch ({_s_ep:.0f}s) ÷ DDP ({_d_ep:.0f}s).")
            else:
                st.info("Missing per-epoch times in the runs to measure the speedup.")

        st.divider()
        subtab_prediction = st.container()

    # ── Report ────────────────────────────────────────────────────────────────
    with subtab_report:
        if not feasibility_csvs:
            st.info("No feasibility CSVs found. Run the analysis from the 'Run analysis' sub-tab.")
        else:
            # ── System profile ─────────────────────────────────────────────────
            st.markdown("### System profile")
            hw_col1, hw_col2, hw_col3, hw_col4 = st.columns(4)
            hw_col1.metric("Model", meta.get("model_name", "—"))
            hw_col2.metric("Parameters (M)", meta.get("total_params_M", "—"))
            hw_col3.metric("GPU", meta.get("hardware_name", "—"))
            hw_col4.metric("Total VRAM (GB)", meta.get("total_vram_gb", "—"))

            # CPU if available
            cpu = meta.get("cpu", {})
            if cpu:
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Logical cores", cpu.get("logical_cores", "—"))
                cc2.metric("Physical cores", cpu.get("physical_cores", "—"))
                cc3.metric("Total RAM (GB)", cpu.get("ram_total_gb", "—"))
                cc4.metric("Free RAM (GB)", cpu.get("ram_free_gb", "—"))

            # Disk if available
            disk = meta.get("disk", {})
            ds_profile = meta.get("dataset", {})
            if disk or ds_profile:
                st.markdown("### Dataset I/O")
                di_cols = st.columns(4)
                if disk:
                    di_cols[0].metric("Disk type", disk.get("type", "—"))
                    di_cols[1].metric("NFS", "Yes" if disk.get("is_nfs") == "yes" else "No")
                    if disk.get("read_mb_per_s", "0") != "0":
                        di_cols[2].metric("Read speed", f"{disk.get('read_mb_per_s', '—')} MB/s")
                        di_cols[3].metric("Patches/s", f"{disk.get('files_per_second', '—')}")
                if ds_profile:
                    io_ratio = float(ds_profile.get("io_bottleneck_ratio", 0) or 0)
                    st.metric("I/O vs compute ratio", f"{io_ratio:.2f}",
                               delta="I/O-bound" if io_ratio > 1.2 else "Compute-bound",
                               delta_color="inverse" if io_ratio > 1.2 else "normal")
                    if io_ratio > 1.2:
                        st.warning("I/O bottleneck: data loading is slower than GPU compute. More GPUs will not improve throughput without a faster disk.")
                    else:
                        st.success("Compute-bound: the GPU is the bottleneck. Adding GPUs (DDP) will speed up training linearly.")

            # Memory by batch size
            mem_keys = ["weight_mb", "gradient_mb", "optimizer_mb", "activation_mb_per_image", "total_static_mb"]
            if any(k in meta for k in mem_keys):
                st.markdown("### Model memory")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Weights (MB)", meta.get("weight_mb", "—"))
                m2.metric("Gradients (MB)", meta.get("gradient_mb", "—"))
                m3.metric("AdamW state (MB)", meta.get("optimizer_mb", "—"))
                m4.metric("Activations/img (MB)", meta.get("activation_mb_per_image", "—"))
                m5.metric("Total static (MB)", meta.get("total_static_mb", "—"))

                # VRAM visual
                total_vram = meta.get("total_vram_gb")
                free_vram = meta.get("free_vram_gb")
                if total_vram and free_vram:
                    fig_vr = go.Figure(go.Bar(
                        x=["Free", "Used"],
                        y=[float(free_vram), float(total_vram) - float(free_vram)],
                        marker_color=[COLORS[2], COLORS[3]], opacity=0.85,
                    ))
                    fig_vr.update_layout(**_base_layout(180, "VRAM distribution"), yaxis_title="GB")
                    _show(fig_vr, "vram_dist")

            # Benchmark
            if not bdf_feas.empty:
                st.markdown("### Throughput benchmark")
                viable = bdf_feas[bdf_feas["oom"] == "no"].copy()
                tp_col = _throughput_col(viable)

                if not viable.empty and tp_col:
                    has_split = ("imgs_per_s_train" in viable.columns and "imgs_per_s_eval" in viable.columns)
                    fig_tp = go.Figure()
                    for mode_feas in viable["trace_mode"].unique():
                        sub = viable[viable["trace_mode"] == mode_feas]
                        x_labels = sub["batch_size"].astype(str) + f" [{mode_feas}]"
                        if has_split:
                            fig_tp.add_trace(go.Bar(x=x_labels, y=sub["imgs_per_s_train"],
                                                    name=f"Train [{mode_feas}]"))
                            fig_tp.add_trace(go.Bar(x=x_labels, y=sub["imgs_per_s_eval"],
                                                    name=f"Eval [{mode_feas}]"))
                        else:
                            fig_tp.add_trace(go.Bar(x=sub["batch_size"].astype(str),
                                                    y=sub[tp_col], name=f"trace={mode_feas}"))
                    fig_tp.update_layout(**_base_layout(300, "Throughput (imgs/s) by batch size"),
                                         barmode="group", xaxis_title="Batch size", yaxis_title="imgs/s")
                    _show(fig_tp, "throughput")

                    if "peak_vram_gb" in viable.columns and viable["peak_vram_gb"].notna().any():
                        fig_vram_f = go.Figure()
                        for mode_feas in viable["trace_mode"].unique():
                            sub = viable[viable["trace_mode"] == mode_feas]
                            fig_vram_f.add_trace(go.Bar(x=sub["batch_size"].astype(str),
                                                        y=sub["peak_vram_gb"], name=f"trace={mode_feas}"))
                        if meta.get("free_vram_gb"):
                            fig_vram_f.add_hline(
                                y=float(meta["free_vram_gb"]), line_dash="dash", line_color="red",
                                annotation_text=f"Free VRAM: {meta['free_vram_gb']} GB",
                                annotation_position="top left",
                            )
                        fig_vram_f.update_layout(**_base_layout(260, "Peak VRAM by batch size"),
                                                  barmode="group", xaxis_title="Batch size", yaxis_title="GB")
                        _show(fig_vram_f, "peak_vram")

                st.dataframe(bdf_feas, use_container_width=True, height=220)
                _dl_csv(bdf_feas, "feasibility_benchmark.csv", "Download benchmark")

                # Time estimates
                est_cols = [c for c in bdf_feas.columns if c.startswith("est_")]
                if est_cols:
                    st.markdown("### Time estimates")
                    orig_ep_col = next(
                        (c for c in bdf_feas.columns if c.startswith("est_total_h_") and c.endswith("ep")), None
                    )
                    orig_n = None
                    if orig_ep_col:
                        try:
                            orig_n = int(orig_ep_col.split("est_total_h_")[1].replace("ep", ""))
                        except ValueError:
                            pass
                    recalc_n = st.number_input("Epochs for total estimate", min_value=1, value=orig_n or 30)
                    display_cols = ["batch_size", "trace_mode", "oom"]
                    for c in ["est_train_min_per_epoch", "est_eval_min_per_epoch", "est_total_min_per_epoch"]:
                        if c in bdf_feas.columns:
                            display_cols.append(c)
                    if orig_ep_col:
                        display_cols.append(orig_ep_col)
                    est_df = bdf_feas[[c for c in display_cols if c in bdf_feas.columns]].copy()
                    per_epoch_col = next((c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                                          if c in bdf_feas.columns), None)
                    if per_epoch_col:
                        est_df[f"est_total_h_{recalc_n}ep"] = (bdf_feas[per_epoch_col] * recalc_n / 60).round(2)
                    st.dataframe(est_df, use_container_width=True)
                    _dl_csv(est_df, "time_estimates.csv", "Download estimates")

        # DDP scenarios (1/2/4/8 GPUs) — section inside the Report
        st.divider()
        subtab_ddp_opt = st.container()

    # ── Real empirical study (mini-training + LR range + gradient noise) ────────
    with subtab_study:
        if not feasibility_csvs:
            st.info("Run the feasibility analysis first.")
        else:
            study = meta.get("study")
            if not study:
                st.info(
                    "This report does not include an empirical study. To generate it, run the "
                    "analysis with `--convergence-study` (real mini-training with LR range test "
                    "and gradient noise scale)."
                )
            else:
                st.markdown("## Empirical convergence study")
                st.caption(
                    "Real measurements on this machine via a short mini-training, "
                    "not extrapolation from historical data."
                )

                # ── LR range test ──────────────────────────────────────────────
                lr_data = study.get("lr", {})
                lr_lrs = study.get("lr_curve_lrs", [])
                lr_losses = study.get("lr_curve_losses", [])
                if lr_data and lr_lrs and lr_losses:
                    st.markdown("### LR range test")
                    sug = float(lr_data.get("suggested_lr", 0) or 0)
                    minl = float(lr_data.get("min_loss_lr", 0) or 0)
                    lr1, lr2, lr3 = st.columns(3)
                    lr1.metric("Suggested LR", f"{sug:.2e}")
                    lr2.metric("Min-loss LR", f"{minl:.2e}")
                    div = lr_data.get("diverged_lr", "")
                    lr3.metric("Divergence LR", f"{float(div):.2e}" if div else "—")

                    fig_lr = go.Figure()
                    fig_lr.add_trace(go.Scatter(
                        x=lr_lrs, y=lr_losses, mode="lines+markers",
                        line=dict(color=COLORS[0], width=2), name="Loss",
                    ))
                    if sug > 0:
                        fig_lr.add_vline(x=sug, line_dash="dash", line_color=COLORS[2],
                                         annotation_text=f"Suggested {sug:.1e}",
                                         annotation_position="top")
                    fig_lr.update_layout(
                        **_base_layout(340, "Loss vs Learning Rate (sweep)"),
                        xaxis_title="Learning rate (log)", yaxis_title="Loss",
                    )
                    fig_lr.update_xaxes(type="log")
                    _show(fig_lr, "lr_range_test")
                    st.caption(
                        "The suggested LR is where the loss drops fastest (the steepest "
                        "negative-slope region), typically ~1 order below the minimum."
                    )

                # ── Curva de convergencia medida ───────────────────────────────
                conv = study.get("conv", {})
                conv_steps = study.get("conv_steps", [])
                conv_losses = study.get("conv_losses", [])
                conv_f1s = study.get("conv_f1s", [])
                if conv and conv_steps:
                    st.markdown("### Measured convergence curve")
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Fit R²", f"{float(conv.get('r_squared', 0) or 0):.3f}")
                    cc2.metric("Estimated Val F1", f"{float(conv.get('best_f1', 0) or 0):.3f}")
                    cc3.metric("Plateau (epoch)", conv.get("epochs_to_plateau", "—"))
                    cc4.metric("Real throughput", f"{float(conv.get('measured_imgs_per_s', 0) or 0):.0f} img/s")

                    # Measured loss curve + extrapolated power-law fit
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        x=conv_steps, y=conv_losses, mode="markers",
                        marker=dict(color=COLORS[0], size=5), name="Measured loss",
                    ))
                    # Fitted curve a·t^-b+c
                    a = float(conv.get("fit_a", 0) or 0)
                    b = float(conv.get("fit_b", 0) or 0)
                    c = float(conv.get("fit_c", 0) or 0)
                    if a > 0 and conv_steps:
                        t_fit = np.linspace(min(conv_steps), max(conv_steps) * 3, 80)
                        y_fit = a * np.power(t_fit, -b) + c
                        fig_conv.add_trace(go.Scatter(
                            x=t_fit, y=y_fit, mode="lines",
                            line=dict(color=COLORS[1], width=2, dash="dash"),
                            name=f"Fit a·t^-b+c (R²={float(conv.get('r_squared',0) or 0):.2f})",
                        ))
                    fig_conv.update_layout(
                        **_base_layout(360, "Measured loss + power-law fit"),
                        xaxis_title="Step", yaxis_title="Loss BCE",
                    )
                    _show(fig_conv, "convergence_loss")

                    # Measured F1 per step
                    if conv_f1s:
                        fig_cf1 = go.Figure(go.Scatter(
                            x=conv_steps, y=conv_f1s, mode="lines+markers",
                            line=dict(color=COLORS[2], width=2), marker=dict(size=4),
                            name="Train F1 (batch)",
                        ))
                        fig_cf1.update_layout(
                            **_base_layout(280, "F1 per step (mini-training)"),
                            xaxis_title="Step", yaxis_title="F1 (batch)",
                        )
                        fig_cf1.update_yaxes(range=[0, 1])
                        _show(fig_cf1, "convergence_f1")

                    st.caption(
                        f"Loss extrapolated to 1 epoch: {float(conv.get('loss_1ep', 0) or 0):.4f} | "
                        f"final: {float(conv.get('loss_final', 0) or 0):.4f}. "
                        "The power-law fit (loss = a·t⁻ᵇ + c) models the initial drop; "
                        "it is extrapolated to the target number of epochs to estimate F1."
                    )

                # ── Gradient noise scale ───────────────────────────────────────
                grad = study.get("grad", {})
                if grad:
                    st.markdown("### Gradient noise scale")
                    gg1, gg2, gg3 = st.columns(3)
                    gg1.metric("Gradient norm",
                               f"{float(grad.get('grad_norm_mean', 0) or 0):.3f} "
                               f"± {float(grad.get('grad_norm_std', 0) or 0):.3f}")
                    gg2.metric("Suggested batch size", grad.get("suggested_batch_size", "—"))
                    gg3.metric("Coeff. of variation", f"{float(grad.get('cv', 0) or 0):.3f}")
                    st.caption(
                        "The gradient noise scale (McCandlish 2018) estimates the critical batch size: "
                        "above it, increasing the batch yields diminishing returns. "
                        "A high CV indicates noisy gradients (suggests a larger batch)."
                    )

    # ── DDP analysis ────────────────────────────────────────────────────────────
    with subtab_ddp_opt:
        if not feasibility_csvs:
            st.info("Run the feasibility analysis first.")
        else:
            st.markdown("## DDP analysis — Optimal resource distribution")
            st.caption(
                "Compares configurations from 1 to 8 GPUs showing batch size, recommended workers, "
                "expected speedup, scaling efficiency and the identified bottleneck."
            )
            ddp_df = parse_ddp_scenarios(meta)

            if ddp_df.empty:
                st.info(
                    "No DDP data in this report. "
                    "Regenerate the analysis with the current version of check_feasibility.py."
                )
            else:
                # ── Scenario table ─────────────────────────────────────────────
                st.markdown("### Scenario table")

                def _color_bottleneck(val: str) -> str:
                    if val == "io":
                        return "background-color: #fee2e2; color: #991b1b"
                    if val == "sync":
                        return "background-color: #fef3c7; color: #92400e"
                    return "background-color: #d1fae5; color: #065f46"

                if "bottleneck" in ddp_df.columns:
                    styled_ddp = ddp_df.style.map(_color_bottleneck, subset=["bottleneck"])
                    if "speedup" in ddp_df.columns:
                        styled_ddp = styled_ddp.background_gradient(
                            subset=["speedup"], cmap="RdYlGn", vmin=1.0, vmax=float(ddp_df["n_gpus"].max() or 8)
                        )
                else:
                    styled_ddp = ddp_df.style
                st.dataframe(styled_ddp, use_container_width=True, hide_index=True)
                _dl_csv(ddp_df, "ddp_scenarios.csv", "Download DDP scenarios")

                # ── Load distribution rectangles ───────────────────────────────
                st.markdown("### Load distribution per GPU")
                st.caption(
                    "Each bar shows the share of batch time: "
                    "compute (green), data I/O (orange), gradient synchronization (red)."
                )

                # Calcular proporciones por GPU
                if {"speedup", "sync_overhead_pct", "n_gpus"}.issubset(ddp_df.columns):
                    viable_ddp = ddp_df[pd.to_numeric(ddp_df["n_gpus"], errors="coerce") > 0].copy()
                    for col in ["sync_overhead_pct", "speedup", "n_gpus"]:
                        viable_ddp[col] = pd.to_numeric(viable_ddp[col], errors="coerce")

                    # Estimate I/O overhead from the ratio if available
                    io_ratio = float(meta.get("dataset", {}).get("io_bottleneck_ratio", 0) or 0)
                    io_pct_est = min(io_ratio * 30, 50)  # Estimate: if ratio=1, I/O ≈ 30% of the time

                    fig_rect = go.Figure()
                    labels = [f"{int(row['n_gpus'])} GPU(s)" for _, row in viable_ddp.iterrows()]
                    sync_pcts = viable_ddp["sync_overhead_pct"].fillna(0).tolist()
                    compute_pcts = [max(0, 100 - s - io_pct_est) for s in sync_pcts]
                    io_pcts = [io_pct_est] * len(labels)

                    fig_rect.add_trace(go.Bar(
                        name="GPU compute", x=labels, y=compute_pcts,
                        marker_color=COLORS[2], opacity=0.85,
                    ))
                    fig_rect.add_trace(go.Bar(
                        name="Data I/O", x=labels, y=io_pcts,
                        marker_color=COLORS[1], opacity=0.85,
                    ))
                    fig_rect.add_trace(go.Bar(
                        name="Gradient sync", x=labels, y=sync_pcts,
                        marker_color=COLORS[3], opacity=0.85,
                    ))
                    fig_rect.update_layout(
                        **_base_layout(360, "Batch time breakdown (%) — estimate"),
                        barmode="stack",
                        xaxis_title="DDP configuration",
                        yaxis_title="Percentage of batch time",
                    )
                    fig_rect.update_yaxes(range=[0, 100])
                    _show(fig_rect, "ddp_load_distribution")

                    # ── Speedup vs theoretical ─────────────────────────────────
                    st.markdown("### Speedup: real vs theoretical")
                    if "speedup" in viable_ddp.columns:
                        n_gpus_vals = viable_ddp["n_gpus"].tolist()
                        speedup_vals = viable_ddp["speedup"].tolist()
                        theoretical = n_gpus_vals  # linear theoretical speedup

                        fig_su = go.Figure()
                        fig_su.add_trace(go.Scatter(
                            x=n_gpus_vals, y=theoretical,
                            name="Theoretical (100% efficiency)",
                            mode="lines+markers", line=dict(color=COLORS[4], width=2, dash="dash"),
                        ))
                        fig_su.add_trace(go.Scatter(
                            x=n_gpus_vals, y=speedup_vals,
                            name="Estimated real speedup",
                            mode="lines+markers",
                            line=dict(color=COLORS[2], width=3),
                            marker=dict(size=10),
                        ))
                        fig_su.update_layout(
                            **_base_layout(320, "Real vs theoretical speedup"),
                            xaxis_title="Number of GPUs",
                            yaxis_title="Speedup",
                        )
                        fig_su.update_xaxes(tickvals=n_gpus_vals)
                        _show(fig_su, "ddp_speedup")

                    # ── Estimated total time per configuration ─────────────────
                    if "time_total_h" in viable_ddp.columns:
                        st.markdown("### Estimated total time per configuration")
                        viable_ddp["time_total_h_num"] = pd.to_numeric(viable_ddp["time_total_h"], errors="coerce")
                        fig_tt = go.Figure(go.Bar(
                            x=labels,
                            y=viable_ddp["time_total_h_num"].tolist(),
                            marker_color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
                            opacity=0.85,
                            text=[f"{v:.1f}h" for v in viable_ddp["time_total_h_num"].tolist()],
                            textposition="outside",
                        ))
                        fig_tt.update_layout(
                            **_base_layout(280, "Total training time (h)"),
                            xaxis_title="DDP configuration",
                            yaxis_title="Hours",
                        )
                        _show(fig_tt, "ddp_total_time")

    # ── F1 performance prediction ───────────────────────────────────────────────
    with subtab_prediction:
        if not feasibility_csvs:
            st.info("Run the feasibility analysis first.")
        else:
            st.markdown("## Empirical performance prediction")
            pred = meta.get("prediction", {})
            curve_val = meta.get("curve_val_f1", [])
            curve_train = meta.get("curve_train_f1", [])
            curve_epochs = meta.get("curve_epochs", [])

            if not pred:
                st.info(
                    "No prediction data in this report. "
                    "Regenerate with the current version of check_feasibility.py."
                )
            else:
                # ── Key prediction metrics ─────────────────────────────────────
                pred_best_f1 = float(pred.get("predicted_best_f1", 0) or 0)
                pred_best_ep = int(float(pred.get("predicted_best_epoch", 0) or 0))
                pred_stop_ep = int(float(pred.get("predicted_early_stop_epoch", 0) or 0))
                confidence = pred.get("confidence", "—")

                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Expected Val F1", f"{pred_best_f1:.3f}")
                pc2.metric("Estimated best epoch", pred_best_ep)
                pc3.metric("Estimated early stop", pred_stop_ep)
                pc4.metric("Confidence", confidence)

                # ── Predicted F1 curve ─────────────────────────────────────────
                if curve_val and curve_epochs:
                    st.markdown("### Estimated F1 curve")
                    st.caption(
                        "Prediction based on historical data from real BigEarthNet-S2 trainings. "
                        "The uncertainty band reflects the observed variability (±0.008 F1 across runs)."
                    )

                    fig_pred = go.Figure()

                    # Uncertainty band
                    uncertainty = 0.015
                    fig_pred.add_trace(go.Scatter(
                        x=curve_epochs + curve_epochs[::-1],
                        y=[v + uncertainty for v in curve_val] + [v - uncertainty for v in curve_val[::-1]],
                        fill="toself", fillcolor="rgba(37,99,235,0.1)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="Uncertainty (±0.015 F1)",
                        showlegend=True,
                    ))

                    # Predicted Val F1
                    fig_pred.add_trace(go.Scatter(
                        x=curve_epochs, y=curve_val,
                        name="Estimated Val F1",
                        mode="lines", line=dict(color=COLORS[0], width=3),
                    ))

                    # Predicted Train F1
                    if curve_train:
                        fig_pred.add_trace(go.Scatter(
                            x=curve_epochs, y=curve_train,
                            name="Estimated Train F1",
                            mode="lines", line=dict(color=COLORS[0], width=2, dash="dot"),
                            opacity=0.6,
                        ))

                    # Mark best epoch
                    if pred_best_ep <= max(curve_epochs):
                        best_val = curve_val[pred_best_ep - 1] if pred_best_ep <= len(curve_val) else pred_best_f1
                        fig_pred.add_trace(go.Scatter(
                            x=[pred_best_ep], y=[best_val],
                            name=f"Best epoch ({pred_best_ep})",
                            mode="markers", marker=dict(color="gold", size=14, symbol="star"),
                        ))

                    # Mark early stop
                    if pred_stop_ep <= max(curve_epochs):
                        fig_pred.add_vline(
                            x=pred_stop_ep, line_dash="dash", line_color=COLORS[3],
                            annotation_text=f"Early stop ~ep{pred_stop_ep}",
                            annotation_position="top right",
                        )

                    # Curva real si hay run seleccionado
                    if selected_run is not None:
                        try:
                            df_actual_pred = _load_df(
                                str(selected_run.log_path),
                                str(selected_run.epoch_csv_path) if selected_run.epoch_csv_path else None,
                            )
                            if not df_actual_pred.empty and "val_f1" in df_actual_pred.columns:
                                fig_pred.add_trace(go.Scatter(
                                    x=df_actual_pred["epoch"].tolist(),
                                    y=df_actual_pred["val_f1"].tolist(),
                                    name="Real Val F1",
                                    mode="lines+markers",
                                    line=dict(color=COLORS[1], width=2.5),
                                    marker=dict(size=5),
                                ))
                        except Exception:
                            pass

                    fig_pred.update_layout(
                        **_base_layout(420, "Validation F1 curve — prediction vs real"),
                        xaxis_title="Epoch",
                        yaxis_title="Val F1 (macro)",
                    )
                    fig_pred.update_yaxes(range=[0.0, 1.0])
                    _show(fig_pred, "f1_prediction")

                    if selected_run is not None:
                        st.caption(
                            "Yellow line = empirical prediction | "
                            "Orange = real data from the selected run | "
                            "Gold star = estimated best epoch"
                        )
                    else:
                        st.caption(
                            "Select a run in the sidebar to overlay the real results."
                        )

                # Prediction data as a downloadable table
                if curve_val and curve_epochs:
                    import pandas as pd
                    pred_curve_df = pd.DataFrame({
                        "epoch": curve_epochs,
                        "val_f1_pred": curve_val,
                        "train_f1_pred": curve_train if curve_train else [None] * len(curve_epochs),
                        "val_f1_upper": [v + 0.015 for v in curve_val],
                        "val_f1_lower": [v - 0.015 for v in curve_val],
                    })
                    _dl_csv(pred_curve_df, "predicted_f1_curve.csv", "Download predicted curve")

    # ── Run analysis ──────────────────────────────────────────────────────────
    with subtab_run_feas:
        st.subheader("Run feasibility analysis")
        configs_available = _get_configs()
        model_options_f = [
            "vit_base_patch16_224", "vit_tiny_patch16_224",
            "vit_small_patch16_224", "resnet50", "efficientnet_b0",
        ]
        with st.form("feasibility_form"):
            fa1, fa2 = st.columns(2)
            with fa1:
                feas_model = st.selectbox("Model", model_options_f)
                feas_batches = st.multiselect("Batch sizes", [16, 32, 64, 128], default=[32, 64])
                feas_epochs = st.number_input("Epochs for estimate", min_value=1, value=30)
                feas_dataset_path = st.text_input(
                    "Dataset path (optional — to measure real I/O)",
                    placeholder="/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2",
                )
            with fa2:
                feas_traces = st.multiselect("Trace modes", ["off", "simple", "deep"],
                                              default=["off", "simple"])
                feas_nfs = st.slider("NFS factor", 1.0, 2.0, 1.0, 0.05,
                                     help="Correction for NFS latency (Verode: ~1.3)")
                feas_config = st.selectbox(
                    "YAML config (optional)",
                    ["(none)"] + (configs_available if configs_available else []),
                )
                feas_no_disk = st.checkbox("Skip I/O measurement (faster)", value=False)
                feas_study = st.checkbox(
                    "Real empirical study (mini-training + LR range + gradient noise)",
                    value=False,
                    help="Measures real convergence on this machine. Slower (~3-8 min).",
                )
                feas_study_steps = st.number_input(
                    "Mini-training steps", min_value=20, max_value=200, value=60,
                    help="Only if the empirical study is enabled",
                )
            submitted_feas = st.form_submit_button("Run")

        if submitted_feas:
            if not feas_batches:
                st.error("Select at least one batch size.")
            else:
                parts = [
                    "uv run python scripts/check_feasibility.py",
                    f"--model {feas_model}",
                    f"--batch-sizes {' '.join(str(b) for b in feas_batches)}",
                    f"--epochs {feas_epochs}",
                    f"--trace-modes {' '.join(feas_traces) if feas_traces else 'off'}",
                ]
                if feas_nfs != 1.0:
                    parts.append(f"--nfs-factor {feas_nfs}")
                if feas_config != "(none)":
                    parts.append(f"--config configs/{feas_config}")
                if feas_dataset_path.strip():
                    parts.append(f'--dataset-path "{feas_dataset_path.strip()}"')
                if feas_no_disk:
                    parts.append("--no-disk-profile")
                if feas_study:
                    parts.append(f"--convergence-study --study-steps {feas_study_steps}")
                cmd = " ".join(parts)
                st.code(cmd, language="bash")
                out_ph = st.empty()
                with st.spinner("Running full analysis…"):
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(ROOT))
                if result.returncode == 0:
                    st.success("Analysis complete.")
                    out_ph.code(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
                    _get_feasibility_csvs.clear()
                else:
                    st.error("Error during the analysis:")
                    out_ph.code(result.stderr[-2000:])

# ═══════════════════════════════════════════════════════════════════════════════
# TIME
# ═══════════════════════════════════════════════════════════════════════════════

with tab_tiempo:
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
        df_time = _load_df(str(run.log_path), str(run.epoch_csv_path) if run.epoch_csv_path else None)

        if "epoch_time" not in df_time.columns or df_time["epoch_time"].isna().all():
            st.info("No per-epoch time data. Use `--trace simple` to generate it.")
        else:
            et = df_time[["epoch", "epoch_time"]].dropna()
            total_s_t = et["epoch_time"].sum()
            avg_s_t = et["epoch_time"].mean()

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Total", _dur_str(total_s_t))
            t2.metric("Average/epoch", f"{avg_s_t/60:.1f} min")
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
                fig_time.add_trace(go.Scatter(x=et["epoch"], y=df_time["epoch_time_train_s"] / 60,
                                              name="Train (min)", mode="lines",
                                              line=dict(color=COLORS[2], width=2, dash="dot")))
            if has_eval_t:
                fig_time.add_trace(go.Scatter(x=et["epoch"], y=df_time["epoch_time_eval_s"] / 60,
                                              name="Eval (min)", mode="lines",
                                              line=dict(color=COLORS[1], width=2, dash="dash")))

            if len(et) >= 2:
                x_arr = et["epoch"].values.astype(float)
                y_arr = et["epoch_time"].values / 60
                coeffs = np.polyfit(x_arr, y_arr, 1)
                fig_time.add_trace(go.Scatter(x=et["epoch"], y=np.polyval(coeffs, x_arr),
                                              name="Trend", mode="lines",
                                              line=dict(color="#94a3b8", width=1, dash="dash")))

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
                fig_time.add_vrect(x0=0.5, x1=warmup_ep + 0.5, fillcolor="#f59e0b", opacity=0.07,
                                   annotation_text=f"Warmup ({warmup_ep} ep)",
                                   annotation_position="top left")

            feasibility_csvs_t = _get_feasibility_csvs()
            # Pick the feasibility whose MODEL matches the displayed run; otherwise
            # the estimate line would be for a different model (a false comparison).
            feas_match_t = None
            for _fc in feasibility_csvs_t:
                try:
                    _m_t, _ = parse_feasibility_csv(_fc)
                    if run.model and _m_t.get("model_name") == run.model:
                        feas_match_t = _fc
                        break
                except Exception:
                    pass
            if feas_match_t is None and feasibility_csvs_t:
                feas_match_t = feasibility_csvs_t[0]
            if feas_match_t:
                try:
                    _, bdf_t = parse_feasibility_csv(feas_match_t)
                    viable_t = bdf_t[bdf_t["oom"] == "no"].copy()
                    tp_col_t = _throughput_col(viable_t)
                    per_ep_col = next((c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                                       if c in viable_t.columns), None)
                    _idx_t = (viable_t[tp_col_t].idxmax()
                               if (tp_col_t and per_ep_col and not viable_t.empty) else None)
                    if _idx_t is not None and not pd.isna(_idx_t):
                        est_min = float(viable_t.loc[_idx_t, per_ep_col])
                        fig_time.add_hline(y=est_min, line_dash="dash", line_color=COLORS[1],
                                           annotation_text=f"Feasibility estimate: {est_min:.0f} min/epoch",
                                           annotation_position="top right")
                except Exception:
                    pass

            fig_time.update_layout(**_base_layout(380, "Time per epoch"),
                                   xaxis_title="Epoch", yaxis_title="Minutes")
            _show(fig_time, "time_per_epoch")
            _dl_csv(et.assign(epoch_time_min=et["epoch_time"] / 60),
                    "time_per_epoch.csv", "Download time data")

            if feasibility_csvs_t:
                try:
                    _, bdf_c = parse_feasibility_csv(feasibility_csvs_t[0])
                    viable_c = bdf_c[bdf_c["oom"] == "no"].copy()
                    tp_col_c = _throughput_col(viable_c)
                    per_ep_c = next((c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                                     if c in viable_c.columns), None)
                    _idx_c = (viable_c[tp_col_c].idxmax()
                               if (tp_col_c and per_ep_c and not viable_c.empty) else None)
                    if _idx_c is not None and not pd.isna(_idx_c):
                        est_val = float(viable_c.loc[_idx_c, per_ep_c])
                        real_val = avg_s_t / 60
                        err_pct = (real_val - est_val) / est_val * 100 if est_val else 0
                        st.markdown("**Estimated vs Real**")
                        ce1, ce2, ce3 = st.columns(3)
                        ce1.metric("Estimated (min/epoch)", f"{est_val:.1f}")
                        ce2.metric("Real average (min/epoch)", f"{real_val:.1f}")
                        ce3.metric("Relative error", f"{err_pct:+.1f}%")
                except Exception:
                    pass

# ═══════════════════════════════════════════════════════════════════════════════
# INFO
# ═══════════════════════════════════════════════════════════════════════════════

with tab_info:
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
        df_info = _load_df(str(run.log_path), str(run.epoch_csv_path) if run.epoch_csv_path else None)
        n_ep_i = len(df_info)
        best_f1_i = _safe_max(df_info["val_f1"]) if "val_f1" in df_info.columns else float("nan")
        best_ep_i_v = _safe_val_at_best(df_info, "val_f1", "epoch")

        col_m, col_f = st.columns(2)

        with col_m:
            st.subheader("Run metadata")
            _cfg_run = _run_config(str(run.log_path))
            rows_i = {
                "Model": run.model or "—",
                "Environment · Mode": f"{run.env} · {run.mode}",
                "Trace mode": run.trace_mode,
                "Epochs": n_ep_i,
                "Best Val F1": f"{best_f1_i:.4f}" if not pd.isna(best_f1_i) else "—",
                "Best epoch": int(best_ep_i_v) if best_ep_i_v is not None else "—",
            }
            # Run config (only new / backfilled runs record it)
            if _cfg_run.get("batch"):
                rows_i["Batch size"] = _cfg_run["batch"]
            if _cfg_run.get("reparto"):
                rows_i["Data split"] = _cfg_run["reparto"]
            if _cfg_run.get("lr"):
                rows_i["Learning rate"] = _cfg_run["lr"]
            if _cfg_run.get("train"):
                rows_i["Train/val images"] = f"{_cfg_run.get('train', '?')} / {_cfg_run.get('val', '?')}"
            if "epoch_time" in df_info.columns and df_info["epoch_time"].notna().any():
                total_si = df_info["epoch_time"].sum()
                rows_i["Total time"] = _dur_str(total_si)
                rows_i["Average/epoch"] = f"{df_info['epoch_time'].mean()/60:.1f} min"
            for k, v in rows_i.items():
                st.markdown(f"**{k}:** {v}")
            if not _cfg_run:
                st.caption("ℹ️ Batch size is only recorded in new runs (from this "
                           "version onwards). Earlier ones did not store it in the log.")

        with col_f:
            st.subheader("Associated files")
            for label_f, path_f in [
                ("Batch CSV", run.batch_csv_path),
                ("Per-class CSV", run.perclass_csv_path),
                ("Epoch CSV", run.epoch_csv_path),
                ("Confusion matrix CSV", run.confusion_matrix_csv_path),
            ]:
                st.markdown(f"- **{label_f}:** `{path_f.name if path_f else '—'}`")

        st.markdown("---")

        st.subheader("YAML config")
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
            st.caption("Could not determine the config for this run.")

        st.subheader("Anomaly detection")
        anomalies = _detect_anomalies(run.log_path)
        if anomalies:
            st.warning(f"{len(anomalies)} line(s) with detected anomalies.")
            with st.expander("View anomalies"):
                for line in anomalies:
                    st.text(line)
        else:
            st.success("No anomalies detected in the log.")

        st.subheader("Log")
        search_term = st.text_input("Filter log lines", "")
        try:
            all_lines = run.log_path.read_text(errors="replace").splitlines()
            if search_term:
                disp_lines = [ln for ln in all_lines if search_term.lower() in ln.lower()]
                st.caption(f"{len(disp_lines)} / {len(all_lines)} lines")
            else:
                disp_lines = all_lines
                st.caption(f"{len(all_lines)} lines total")
            st.code("\n".join(disp_lines[-400:]), language="text")
        except Exception as exc:
            st.error(str(exc))

# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_lanzador:
    subtab_single, subtab_ddp_l = st.tabs(["Single GPU", "DDP (multi-GPU)"])

    configs_l = _get_configs()
    model_opts_l = [
        "vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224",
        "resnet50", "efficientnet_b0", "deit_tiny_patch16_224",
    ]

    with subtab_single:
        st.subheader("Single-GPU training")
        with st.form("launcher_single_form"):
            la1, la2 = st.columns(2)
            with la1:
                l_model = st.selectbox("Model", model_opts_l)
                l_config = st.selectbox("YAML config", configs_l if configs_l else ["(none)"])
                l_epochs = st.number_input("Override epochs", min_value=0, value=0,
                                           help="0 = use the config value")
                l_batch = st.number_input("Override batch size", min_value=0, value=0,
                                          help="0 = use the config value")
            with la2:
                l_trace = st.selectbox("Trace mode", ["simple", "off", "deep"])
                l_layers = st.multiselect("Layers", ["plot", "hooks", "confusion", "batch-monitor"],
                                          default=["confusion"])
                l_fn = st.multiselect("Fn decorators", ["timing", "energy"])
                l_inspect = st.multiselect("Inspect features",
                                           ["model-summary", "grad-monitor", "anomalies", "batch-table"])
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

    with subtab_ddp_l:
        st.subheader("DDP training")
        with st.form("launcher_ddp_form"):
            dd1, dd2 = st.columns(2)
            with dd1:
                d_nproc = st.number_input("GPUs (--nproc_per_node)", min_value=1, max_value=8, value=2)
                d_model = st.selectbox("Model", model_opts_l, key="ddp_model")
                d_config = st.selectbox("YAML config", configs_l if configs_l else ["(none)"],
                                         key="ddp_config")
                d_epochs = st.number_input("Override epochs", min_value=0, value=0, key="ddp_ep")
            with dd2:
                d_trace = st.selectbox("Trace mode", ["simple", "off", "deep"], key="ddp_trace")
                d_layers = st.multiselect("Layers", ["plot", "confusion", "batch-monitor"],
                                          default=["confusion"], key="ddp_layers")
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
# LIVE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_envivo:
    st.subheader("Live monitor")

    now_ts = time.time()
    recent_runs = [
        r for r in runs
        if r.log_path.exists() and (now_ts - r.log_path.stat().st_mtime) < 1800
    ]

    if not recent_runs:
        st.info(
            "No active runs (no log modified in the last 30 min). "
            "Launch a training from the Launcher tab."
        )
    else:
        live_labels = {r.label: r for r in recent_runs}
        live_sel = st.selectbox("Active run", list(live_labels.keys()), key="live_run_sel")
        live_run = live_labels[live_sel]

        @st.fragment(run_every=refresh_interval)
        def _live_panel(run: RunInfo):
            _load_df.clear()

            gpu = _gpu_usage()
            if gpu:
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("GPU", gpu["name"])
                g2.metric("VRAM", f"{gpu['mem_used_mb']/1024:.1f} / {gpu['mem_total_mb']/1024:.1f} GB")
                g3.metric("Utilization", f"{gpu['util_pct']}%")
                g4.metric("Temperature", f"{gpu['temp_c']} °C")
            else:
                st.caption("GPU info unavailable (nvidia-smi not found).")

            progress = _parse_log_progress(run.log_path)
            if progress["epochs"] > 0:
                pct = progress["epoch"] / progress["epochs"]
                st.progress(pct, text=f"Epoch {progress['epoch']} / {progress['epochs']}")

            if progress["last_val_f1"] is not None:
                m1, m2 = st.columns(2)
                m1.metric("Last Val F1", f"{progress['last_val_f1']:.4f}")
                if progress["last_val_loss"] is not None:
                    m2.metric("Last Val Loss", f"{progress['last_val_loss']:.4f}")

            if run.epoch_csv_path and run.epoch_csv_path.exists():
                live_df = _load_df(str(run.log_path), str(run.epoch_csv_path))
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
                    fig_live.update_layout(**_base_layout(280, "Metrics"), xaxis_title="Epoch")
                    _show(fig_live, "live_metrics")

            st.subheader("Log tail")
            st.code(_read_log_tail(run.log_path, n=40), language="text")

        _live_panel(live_run)
