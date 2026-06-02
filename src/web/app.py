"""Streamlit web dashboard — Training Dashboard v6 (español)."""

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

# ── Configuración de página ────────────────────────────────────────────────────

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

# ── Cargadores con caché ───────────────────────────────────────────────────────


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


# ── Cargadores de dataset con caché ─────────────────────────────────────────────


@st.cache_data(ttl=600)
def _load_class_distribution(parquet_str: str) -> pd.DataFrame | None:
    """Distribución de clases cacheada (itera ~237K filas, lento)."""
    return class_distribution_from_parquet(Path(parquet_str))


@st.cache_data(ttl=600)
def _load_example_images(parquet_str: str, root_str: str, class_name: str, n: int = 4):
    """Carga n imágenes RGB de ejemplo para una clase, cacheadas."""
    patches = find_example_patches(Path(parquet_str), class_name, n=n)
    images = []
    for pid in patches:
        img = load_rgb_image(Path(root_str), pid)
        if img is not None:
            images.append((pid, img))
    return images


# ── Helpers generales ──────────────────────────────────────────────────────────


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


# ── Helpers de gráficas ────────────────────────────────────────────────────────

_PLOTLY_CFG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def _show(fig: go.Figure, key: str | None = None) -> None:
    """Muestra una gráfica Plotly con barra de herramientas visible y descarga PNG."""
    cfg = dict(_PLOTLY_CFG)
    if key:
        cfg["toImageButtonOptions"] = {"format": "png", "scale": 2, "filename": key}
    st.plotly_chart(fig, use_container_width=True, config=cfg)


def _dl_csv(df: pd.DataFrame, filename: str = "datos.csv", label: str = "Descargar CSV") -> None:
    """Botón de descarga de un DataFrame como CSV."""
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


# ── Grupos de clases (para confusion matrix) ──────────────────────────────────

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
        st.warning("No se encontraron runs en logs/.")
        selected_run = None
        run = None
    else:
        trace_filter = st.selectbox("Trace mode", ["todos", "simple", "deep"])
        filtered = [r for r in runs if trace_filter == "todos" or r.trace_mode == trace_filter]

        if not filtered:
            st.warning("No hay runs para este filtro.")
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
                f"**Entorno:** {run.env}  \n"
                f"**Modo:** {run.mode}  \n"
                f"**Modelo:** {run.model or '—'}  \n"
                f"**Trace:** {run.trace_mode}  \n"
                f"**Epoch CSV:** {'sí' if has_csv else 'no'}  \n"
                f"**Batch CSV:** {'sí' if run.batch_csv_path else 'no'}  \n"
                f"**Per-class CSV:** {'sí' if run.perclass_csv_path else 'no'}"
            )

    st.markdown("---")
    st.markdown("**Monitor en vivo**")
    refresh_interval = st.slider("Intervalo de refresco (s)", 5, 60, 10)

# ── Pestañas ──────────────────────────────────────────────────────────────────

(
    tab_inicio, tab_sistema, tab_dataset, tab_modelos,
    tab_curvas, tab_porclase, tab_batch, tab_comparar, tab_ddp,
    tab_viabilidad, tab_tiempo, tab_info,
    tab_lanzador, tab_envivo,
) = st.tabs([
    "Inicio", "Sistema", "Dataset", "Modelos",
    "Curvas", "Por clase", "Batch", "Comparar", "Análisis DDP",
    "Viabilidad", "Tiempo", "Información", "Lanzador", "En vivo",
])

# ═══════════════════════════════════════════════════════════════════════════════
# INICIO — pantalla principal con cuadrícula de resumen
# ═══════════════════════════════════════════════════════════════════════════════

with tab_inicio:
    st.markdown("## Vista general del proyecto")

    # ── Estadísticas globales ──────────────────────────────────────────────────
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
    g2.metric("Mejor Val F1", f"{best_f1_global:.4f}" if best_f1_global > float("-inf") else "—")
    g3.metric("Run destacado", best_run_label[:28] if best_run_label != "—" else "—")
    g4.metric("Tiempo GPU total", f"{total_gpu_h:.1f} h")
    g5.metric("Reportes viabilidad", len(feasibility_csvs_home))

    st.markdown("---")

    # ── Run seleccionado: resumen + mini curvas ────────────────────────────────
    if selected_run is not None:
        st.markdown(f"### Run seleccionado — `{selected_run.label}`")
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
                m1.metric("Epochs completados", n_ep_sel)
                m2.metric("Mejor Val F1", f"{best_f1_sel:.4f}" if not pd.isna(best_f1_sel) else "—")
                m3, m4 = st.columns(2)
                m3.metric("Mejor epoch", int(best_ep_v) if best_ep_v is not None else "—")
                m4.metric("Duración", dur_sel or "—")
                if thresh_f1 is not None:
                    st.metric("F1 @ threshold óptimo", f"{thresh_f1:.4f}")
                anomalies_home = _detect_anomalies(selected_run.log_path)
                if anomalies_home:
                    st.warning(f"{len(anomalies_home)} anomalía(s) en el log")
                else:
                    st.success("Sin anomalías detectadas")
            else:
                st.info("No hay datos de métricas para este run.")

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

    # ── Estado del sistema ─────────────────────────────────────────────────────
    st.markdown("### Estado del sistema")
    gpu_home = _gpu_usage()

    if gpu_home:
        # Nombre completo de la GPU como encabezado (evita el truncado en st.metric)
        gpu_name_clean = gpu_home["name"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        st.markdown(f"**GPU:** {gpu_home['name']}")
        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Modelo", gpu_name_clean)
        gc2.metric("VRAM", f"{gpu_home['mem_used_mb']/1024:.1f} / {gpu_home['mem_total_mb']/1024:.1f} GB")
        gc3.metric("Utilización GPU", f"{gpu_home['util_pct']}%")
        gc4.metric("Temperatura", f"{gpu_home['temp_c']} °C")
    else:
        st.caption("GPU: nvidia-smi no disponible")

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

    # ── Snapshot por clase ─────────────────────────────────────────────────────
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
                st.markdown(f"### Rendimiento por clase (epoch {last_ep_home})")
                ph_left, ph_right = st.columns(2)
                with ph_left:
                    st.markdown("**Top 5 mejores clases**")
                    top5 = ep_pc_home.head(5)[["class_name", "f1", "precision", "recall"]]
                    st.dataframe(
                        top5.style.map(_color_f1_cell, subset=["f1"])
                            .format({"f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"}),
                        use_container_width=True, hide_index=True,
                    )
                with ph_right:
                    st.markdown("**Top 5 peores clases**")
                    bot5 = ep_pc_home.tail(5).sort_values("f1")[["class_name", "f1", "precision", "recall"]]
                    st.dataframe(
                        bot5.style.map(_color_f1_cell, subset=["f1"])
                            .format({"f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"}),
                        use_container_width=True, hide_index=True,
                    )
                st.markdown("---")
        except Exception:
            pass

    # ── Tabla de todos los runs ────────────────────────────────────────────────
    st.markdown("### Todos los runs")
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
                "Entorno": r.env,
                "Modelo": r.model or "—",
                "Trace": r.trace_mode,
                "Epochs": len(df_r),
                "Mejor Val F1": round(run_best_f1, 4),
                "Mejor epoch": int(best_ep_v) if best_ep_v is not None else "—",
                "Duración": _dur_str(dur_s) if not pd.isna(dur_s) else "—",
                "Energía eval (Wh)": f"{energy_wh:.0f}" if energy_wh else "—",
            })
        except Exception:
            pass

    if overview_rows:
        ov_df = pd.DataFrame(overview_rows)
        st.dataframe(
            ov_df.style.background_gradient(subset=["Mejor Val F1"], cmap="RdYlGn", vmin=0.4, vmax=0.75),
            use_container_width=True, hide_index=True,
        )
        _dl_csv(ov_df, "resumen_runs.csv", "Descargar tabla de runs")
    else:
        st.info("No se encontraron runs con métricas parseables.")

# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA
# ═══════════════════════════════════════════════════════════════════════════════

with tab_sistema:
    st.markdown("## Monitor del sistema")
    ref_int = st.sidebar.slider("Refresco sistema (s)", 2, 30, 5, key="sys_ref_int")

    @st.fragment(run_every=ref_int)
    def _system_panel():
        snap = get_snapshot(disk_paths=["/", "/home", "/media"])

        st.markdown("### CPU")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Uso", f"{snap.cpu.usage_pct:.1f}%")
        sc2.metric("Núcleos lógicos", snap.cpu.count_logical)
        sc3.metric("Núcleos físicos", snap.cpu.count_physical)
        sc4.metric("Frecuencia", f"{snap.cpu.freq_mhz:.0f} MHz" if snap.cpu.freq_mhz else "—")
        st.progress(snap.cpu.usage_pct / 100)

        st.markdown("### RAM")
        sr1, sr2, sr3, sr4 = st.columns(4)
        sr1.metric("Usada", f"{snap.ram.used_gb:.1f} GB")
        sr2.metric("Total", f"{snap.ram.total_gb:.1f} GB")
        sr3.metric("Disponible", f"{snap.ram.available_gb:.1f} GB")
        sr4.metric("Uso %", f"{snap.ram.percent:.1f}%")
        st.progress(snap.ram.percent / 100)
        if snap.ram.swap_total_gb > 0:
            st.caption(f"Swap: {snap.ram.swap_used_gb:.1f} / {snap.ram.swap_total_gb:.1f} GB")

        st.markdown("### GPU")
        if snap.gpus:
            for gpu in snap.gpus:
                mem_pct = gpu.mem_used_mb / gpu.mem_total_mb * 100 if gpu.mem_total_mb else 0
                g1, g2, g3, g4, g5 = st.columns(5)
                g1.metric(f"GPU {gpu.index}", gpu.name[:28])
                g2.metric("VRAM usada", f"{gpu.mem_used_mb / 1024:.1f} GB")
                g3.metric("VRAM total", f"{gpu.mem_total_mb / 1024:.1f} GB")
                g4.metric("Utilización", f"{gpu.util_pct}%")
                g5.metric("Temperatura", f"{gpu.temp_c}°C")
                st.progress(mem_pct / 100,
                            text=f"VRAM {gpu.mem_used_mb}/{gpu.mem_total_mb} MB ({mem_pct:.1f}%)")
                if gpu.power_w is not None:
                    limit_str = f" / {gpu.power_limit_w:.0f} W" if gpu.power_limit_w else ""
                    st.caption(f"Consumo: {gpu.power_w:.1f} W{limit_str}")
        else:
            st.info("No se detectó GPU (nvidia-smi no disponible).")

        st.markdown("### Disco")
        if snap.disks:
            disk_cols = st.columns(len(snap.disks))
            for col, disk in zip(disk_cols, snap.disks):
                col.metric(disk.path, f"{disk.free_gb:.1f} GB libres")
                col.progress(disk.percent / 100,
                             text=f"{disk.used_gb:.1f} / {disk.total_gb:.1f} GB ({disk.percent:.1f}%)")
        else:
            st.info("No se pudo leer el uso de disco.")

        st.markdown("### Red (acumulado desde el arranque)")
        nn1, nn2 = st.columns(2)
        nn1.metric("Enviado", f"{snap.network.bytes_sent_mb / 1024:.2f} GB")
        nn2.metric("Recibido", f"{snap.network.bytes_recv_mb / 1024:.2f} GB")

    _system_panel()

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dataset:
    st.markdown("## Explorador de datos — BigEarthNet-S2 v2.0")

    meta_path: Path | None = None
    for candidate in [
        "/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet",
        "/home/bejeque/alu0101317038/datasets/bigearthnet/metadata.parquet",
    ]:
        if Path(candidate).exists():
            meta_path = Path(candidate)
            break

    st.markdown("### Splits del dataset")
    split_col, pie_col = st.columns([1, 1])
    with split_col:
        ds1, ds2 = st.columns(2)
        ds1.metric("Train", f"{SPLIT_SIZES['train']:,}")
        ds2.metric("Validación", f"{SPLIT_SIZES['val']:,}")
        ds3, ds4 = st.columns(2)
        ds3.metric("Test", f"{SPLIT_SIZES['test']:,}")
        ds4.metric("Total patches", f"{sum(SPLIT_SIZES.values()):,}")
    with pie_col:
        fig_splits = go.Figure(go.Pie(
            labels=["Train", "Validación", "Test"],
            values=list(SPLIT_SIZES.values()),
            hole=0.45,
            marker_colors=[COLORS[0], COLORS[2], COLORS[1]],
            textinfo="label+percent",
        ))
        fig_splits.update_layout(
            **_base_layout(260, "Distribución de splits", margin=dict(l=10, r=10, t=40, b=10)),
            showlegend=False,
        )
        _show(fig_splits, "splits")

    # ── Distribución de clases ─────────────────────────────────────────────────
    st.markdown("### Distribución de clases (split de train)")
    dist_df = None
    if meta_path:
        dist_df = _load_class_distribution(str(meta_path))
        if dist_df is not None:
            st.caption(f"Fuente: {meta_path.name} (conteo real de etiquetas multi-label)")
        else:
            st.caption("No se pudo leer el parquet — usando estadísticas aproximadas.")
    if dist_df is None:
        dist_df = class_distribution_approximate()
        if not meta_path:
            st.caption("metadata.parquet no encontrado — usando estadísticas aproximadas.")

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
        **_base_layout(620, "Muestras por clase (train)", margin=dict(l=300, r=90, t=40, b=40)),
        xaxis_title="Número de muestras (multi-label)", yaxis_title="",
    )
    fig_dist.update_yaxes(tickfont=dict(size=11), automargin=True)
    max_x = dist_df["train_count"].max()
    fig_dist.update_xaxes(range=[0, max_x * 1.15])
    _show(fig_dist, "distribucion_clases")
    st.caption(
        "Rojo = clase rara (<10K), Naranja = moderada (<40K), Verde = frecuente. "
        "Al ser multi-label, la suma de etiquetas supera el número de patches. "
        "Las clases raras limitan el techo de F1 macro."
    )

    st.markdown("### Desbalance de clases")
    max_c = dist_df["train_count"].max()
    min_c = dist_df["train_count"].min()
    ratio = max_c / min_c if min_c > 0 else float("inf")
    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Clase más frecuente", dist_df.iloc[-1]["class"][:28],
               f"{int(max_c):,}")
    ci2.metric("Clase más rara", dist_df.iloc[0]["class"][:28],
               f"{int(min_c):,}")
    ci3.metric("Ratio de desbalance", f"{ratio:.1f}×")
    _dl_csv(dist_df[["class", "train_count"]], "distribucion_clases.csv",
            "Descargar distribución")

    # ── Imágenes de ejemplo por clase ──────────────────────────────────────────
    st.markdown("### Imágenes de ejemplo por clase")
    if not meta_path:
        st.info("Requiere acceso al dataset (metadata.parquet no encontrado).")
    else:
        # Ruta al directorio raíz del dataset (junto al parquet)
        ds_root = meta_path.parent / "BigEarthNet-S2"
        if not ds_root.exists():
            st.info(f"Directorio del dataset no encontrado en {ds_root}.")
        else:
            sel_class = st.selectbox(
                "Clase", CLASS_NAMES,
                index=CLASS_NAMES.index("Marine waters") if "Marine waters" in CLASS_NAMES else 0,
            )
            with st.spinner("Cargando imágenes RGB del dataset…"):
                examples = _load_example_images(str(meta_path), str(ds_root), sel_class, n=4)
            if examples:
                st.caption(
                    "Proxy RGB (bandas Sentinel-2 B04/B03/B02) con stretch de percentiles "
                    "para visibilidad. Cada patch es 120×120 px (~1.2 km²)."
                )
                img_cols = st.columns(len(examples))
                for col, (pid, img) in zip(img_cols, examples):
                    col.image(img, caption=pid.split("_")[-2] + "_" + pid.split("_")[-1],
                              use_container_width=True)
            else:
                st.warning(
                    f"No se pudieron cargar imágenes para '{sel_class}'. "
                    "¿Está el dataset completo y accesible?"
                )

    # ── País ───────────────────────────────────────────────────────────────────
    if meta_path:
        country_counts = get_country_distribution(meta_path)
        if country_counts is not None and not country_counts.empty:
            st.markdown("### Distribución por país (train)")
            top_n = country_counts.head(15).sort_values(ascending=True)
            fig_c = go.Figure(go.Bar(
                x=top_n.values, y=top_n.index, orientation="h",
                marker_color=COLORS[0], opacity=0.85,
                text=[f"{v:,}" for v in top_n.values], textposition="outside",
                cliponaxis=False,
            ))
            fig_c.update_layout(
                **_base_layout(420, "Top 15 países por número de patches",
                               margin=dict(l=120, r=80, t=40, b=40)),
                xaxis_title="Patches", yaxis_title="",
            )
            fig_c.update_xaxes(range=[0, top_n.values.max() * 1.15])
            _show(fig_c, "paises")

    # ── Dificultad vs frecuencia ───────────────────────────────────────────────
    st.markdown("### Dificultad por clase vs frecuencia")
    st.caption("Cruza la frecuencia de cada clase con su F1 de validación (CSV per-class más reciente).")
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
                labels={"train_count": "Muestras de entrenamiento", "f1": "Val F1"},
                title=f"F1 vs frecuencia de clase (epoch {last_ep})",
            )
            fig_sc.update_traces(textposition="top center", textfont_size=9)
            fig_sc.update_layout(
                **_base_layout(480, margin=dict(l=60, r=40, t=40, b=50)),
                showlegend=False,
            )
            fig_sc.update_yaxes(range=[-0.05, 1.05])
            _show(fig_sc, "f1_vs_frecuencia")
            st.caption(
                "Las clases con pocas muestras tienden a tener F1 bajo. "
                "Los puntos abajo a la izquierda son los más difíciles de mejorar."
            )
    else:
        st.info("No se encontró CSV de per-class. Ejecuta un training con `--layers confusion`.")

# ═══════════════════════════════════════════════════════════════════════════════
# MODELOS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_modelos:
    st.markdown("## Explorador de modelos")
    st.caption("Explora modelos timm y compara parámetros y requisitos de VRAM.")

    col_fam, col_bs = st.columns([3, 1])
    with col_fam:
        selected_families = st.multiselect(
            "Familias de modelos", ALL_FAMILIES, default=["ViT", "ResNet", "EfficientNet"],
        )
    with col_bs:
        cmp_batch = st.selectbox("Batch size para estimación VRAM", [4, 8, 16, 32, 64, 128], index=3)

    candidate_models = []
    for fam in selected_families:
        candidate_models.extend(CURATED_MODELS.get(fam, []))

    extra_model = st.text_input("Añadir modelo timm personalizado", placeholder="p.ej. convnext_large")
    if extra_model.strip():
        candidate_models.append(extra_model.strip())

    if not candidate_models:
        st.info("Selecciona al menos una familia.")
    else:
        with st.spinner("Cargando estadísticas de modelos…"):
            rows = compare_models(candidate_models, [cmp_batch], num_classes=19)

        if not rows:
            st.warning("No se pudo cargar ningún modelo.")
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
            st.caption("Verde = cabe en 4 GB | Naranja = 4–8 GB | Rojo = >8 GB (límite RTX 3060 Ti)")
            _dl_csv(cmp_df, "comparativa_modelos.csv", "Descargar comparativa de modelos")

            st.markdown("### Parámetros vs FLOPs")
            plot_df = cmp_df[cmp_df["FLOPs (MFLOPs)"] != "—"].copy()
            if not plot_df.empty:
                plot_df["FLOPs (MFLOPs)"] = pd.to_numeric(plot_df["FLOPs (MFLOPs)"], errors="coerce")
                plot_df[vram_col] = pd.to_numeric(plot_df.get(vram_col, 0), errors="coerce").fillna(1)
                fig_bubble = px.scatter(
                    plot_df, x="FLOPs (MFLOPs)", y="Params (M)",
                    size=vram_col, color="Family", text="Model", hover_name="Model",
                    size_max=40,
                    labels={"FLOPs (MFLOPs)": "FLOPs por imagen (MFLOPs)", "Params (M)": "Parámetros (M)"},
                )
                fig_bubble.update_traces(textposition="top center", textfont_size=8)
                fig_bubble.update_layout(**_base_layout(420, "Complejidad de modelos"), showlegend=True)
                _show(fig_bubble, "complejidad_modelos")
                st.caption("Tamaño de burbuja = VRAM estimada al batch size seleccionado.")

            st.markdown("### VRAM requerida por batch size")
            vram_models = candidate_models[:8]
            with st.spinner("Calculando estimaciones de VRAM…"):
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
                    **_base_layout(380, "VRAM estimada (GB) por batch size"),
                    barmode="group", xaxis_tickangle=30, yaxis_title="GB",
                )
                _show(fig_vram_m, "vram_por_batch")

            st.markdown("### Lanzamiento rápido")
            selected_for_launch = st.selectbox(
                "Selecciona modelo para prellenar el Lanzador", [r["Model"] for r in rows]
            )
            if selected_for_launch:
                st.info(f"Ve a la pestaña **Lanzador** y usa el modelo `{selected_for_launch}`.")
                st.session_state["preselected_model"] = selected_for_launch

# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS DDP
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ddp:
    st.markdown("## Análisis DDP — Single-GPU vs Distribuido")
    st.caption("Compara runs single-GPU y DDP del mismo modelo para medir speedup real, eficiencia y escalabilidad.")

    all_runs_ddp = _get_runs()
    if not all_runs_ddp:
        st.info("No se encontraron runs.")
    else:
        single_runs = [r for r in all_runs_ddp if r.mode == "single"]
        ddp_runs = [r for r in all_runs_ddp if r.mode == "ddp"]

        da1, da2, da3 = st.columns(3)
        da1.metric("Runs single-GPU", len(single_runs))
        da2.metric("Runs DDP", len(ddp_runs))
        da3.metric("Total runs", len(all_runs_ddp))

        if not ddp_runs:
            st.info("No hay runs DDP todavía. Lanza `scripts/train_ddp.py` — los resultados aparecerán aquí automáticamente.")
        else:
            st.markdown("### Runs DDP")
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
                        "Run": r.label[:50], "Modelo": r.model or "—",
                        "Entorno": r.env, "Mejor Val F1": round(best_f1, 4),
                        "Epochs": len(ddf),
                        "Epoch promedio (min)": round(avg_ep / 60, 1) if avg_ep else "—",
                    })
                except Exception:
                    pass
            if ddp_rows:
                st.dataframe(pd.DataFrame(ddp_rows), use_container_width=True, hide_index=True)

        if single_runs and ddp_runs:
            st.markdown("### Análisis de speedup")
            col_s, col_d = st.columns(2)
            with col_s:
                single_lbl = st.selectbox("Run single-GPU", [r.label for r in single_runs], key="ddp_single_sel")
            with col_d:
                ddp_lbl = st.selectbox("Run DDP", [r.label for r in ddp_runs], key="ddp_ddp_sel")

            r_single = next(r for r in single_runs if r.label == single_lbl)
            r_ddp = next(r for r in ddp_runs if r.label == ddp_lbl)

            df_s = _load_df(str(r_single.log_path), str(r_single.epoch_csv_path) if r_single.epoch_csv_path else None)
            df_d = _load_df(str(r_ddp.log_path), str(r_ddp.epoch_csv_path) if r_ddp.epoch_csv_path else None)

            if not df_s.empty and not df_d.empty:
                avg_s = df_s["epoch_time"].mean() if "epoch_time" in df_s.columns and df_s["epoch_time"].notna().any() else None
                avg_d = df_d["epoch_time"].mean() if "epoch_time" in df_d.columns and df_d["epoch_time"].notna().any() else None

                su1, su2, su3, su4 = st.columns(4)
                su1.metric("Epoch single-GPU", f"{avg_s/60:.1f} min" if avg_s else "—")
                su2.metric("Epoch DDP", f"{avg_d/60:.1f} min" if avg_d else "—")
                if avg_s and avg_d and avg_d > 0:
                    speedup = avg_s / avg_d
                    su3.metric("Speedup real", f"{speedup:.2f}×")
                    su4.metric("Eficiencia de escalado", f"{speedup / 2 * 100:.1f}%")

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
                    fig_time_ddp.update_layout(**_base_layout(260, "Tiempo por epoch: Single-GPU vs DDP"),
                                               xaxis_title="Epoch", yaxis_title="Minutos")
                    _show(fig_time_ddp, "ddp_tiempo")

                st.markdown("### Escalado teórico vs real")
                world_sizes = [1, 2, 4, 8]
                if avg_s:
                    theoretical = [avg_s / ws for ws in world_sizes]
                    fig_scale = go.Figure()
                    fig_scale.add_trace(go.Scatter(
                        x=world_sizes, y=[t / 60 for t in theoretical],
                        name="Teórico (100% eficiencia)",
                        line=dict(color=COLORS[4], width=2, dash="dash"), mode="lines+markers",
                    ))
                    if avg_d:
                        fig_scale.add_trace(go.Scatter(
                            x=[2], y=[avg_d / 60], name="DDP real (2 GPUs)",
                            mode="markers", marker=dict(color=COLORS[2], size=14, symbol="star"),
                        ))
                    fig_scale.update_layout(**_base_layout(300, "Tiempo por epoch vs número de GPUs"),
                                            xaxis_title="Número de GPUs", yaxis_title="Minutos por epoch")
                    fig_scale.update_xaxes(tickvals=world_sizes)
                    _show(fig_scale, "escalado_ddp")
                    st.caption("La diferencia entre teórico y real refleja overhead de comunicación, cuello de botella NFS y desequilibrio de carga.")

# ═══════════════════════════════════════════════════════════════════════════════
# CURVAS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_curvas:
    if selected_run is None:
        st.info("Selecciona un run en la barra lateral.")
    else:
        df = _load_df(
            str(run.log_path),
            str(run.epoch_csv_path) if run.epoch_csv_path else None,
        )

        if df.empty:
            st.error("No se pudo parsear ningún epoch del run seleccionado.")
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
            c2.metric("Mejor Val F1", f"{best_f1:.4f}" if not pd.isna(best_f1) else "—")
            c3.metric("Mejor epoch", best_epoch)
            if best_thresh_f1 is not None:
                c4.metric("F1 @ threshold óptimo", f"{best_thresh_f1:.4f}")
            if "epoch_time" in df.columns and df["epoch_time"].notna().any():
                c5.metric("Duración total", _dur_str(df["epoch_time"].sum()))

            src = "epoch_metrics CSV" if (run.epoch_csv_path and run.epoch_csv_path.exists()) else "fichero de log"
            st.caption(f"Fuente: {src}")

            extra_thresh: list = []
            if "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any():
                extra_thresh = [go.Scatter(
                    x=df["epoch"], y=df["f1_at_threshold"],
                    name="F1 @ threshold óptimo", mode="lines",
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
                fig_et.update_layout(**_base_layout(240, "Tiempo por epoch (min)"),
                                     xaxis_title="Epoch", yaxis_title="Minutos")
                _show(fig_et, "tiempo_epoch")

            has_energy = "energy_eval_wh" in df.columns and df["energy_eval_wh"].notna().any()
            if has_energy:
                st.markdown("#### Consumo energético")
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
                        fig_e.update_layout(**_base_layout(260, "Energía por epoch (Wh)"),
                                            barmode="group", xaxis_title="Epoch", yaxis_title="Wh")
                        _show(fig_e, "energia")
                with e2:
                    power_cols = []
                    if "power_eval_w" in df.columns and df["power_eval_w"].notna().any():
                        power_cols.append(("Potencia eval (W)", "power_eval_w", COLORS[1]))
                    if "power_train_w" in df.columns and df["power_train_w"].notna().any():
                        power_cols.append(("Potencia train (W)", "power_train_w", COLORS[0]))
                    if power_cols:
                        fig_p = go.Figure()
                        for name_p, col_p, color_p in power_cols:
                            fig_p.add_trace(go.Scatter(x=df["epoch"], y=df[col_p],
                                                       name=name_p, mode="lines+markers",
                                                       line=dict(color=color_p, width=2)))
                        fig_p.update_layout(**_base_layout(260, "Potencia GPU promedio por epoch (W)"),
                                            xaxis_title="Epoch", yaxis_title="Vatios")
                        _show(fig_p, "potencia_gpu")

                total_eval_wh = df["energy_eval_wh"].sum()
                total_train_wh = (df["energy_train_j"].sum() / 3600
                                  if "energy_train_j" in df.columns and df["energy_train_j"].notna().any() else 0)
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("Energía eval total", f"{total_eval_wh:.1f} Wh")
                if total_train_wh > 0:
                    ec2.metric("Energía train total", f"{total_train_wh:.1f} Wh")
                    ec3.metric("Energía total", f"{total_eval_wh + total_train_wh:.1f} Wh")

            _dl_csv(df, "epoch_metrics.csv", "Descargar epoch_metrics.csv")

            with st.expander("Tabla de epochs completa"):
                st.dataframe(df.set_index("epoch"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# POR CLASE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_porclase:
    if selected_run is None:
        st.info("Selecciona un run en la barra lateral.")
    else:
        subtab_bars, subtab_trend, subtab_cm = st.tabs(
            ["Por clase", "Tendencia", "Matriz de confusión"]
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
                        f"perclass_ep{selected_ep}.csv", "Descargar tabla por clase")

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
                    title=dict(text=f"Métricas por clase — Epoch {selected_ep}", font=dict(size=13)),
                    xaxis_title="Puntuación", xaxis=dict(range=[0, 1]),
                    height=600, margin=dict(l=200, r=16, t=36, b=40),
                    paper_bgcolor="white", plot_bgcolor="#f8fafc",
                )
                _show(fig_pc, f"por_clase_ep{selected_ep}")
            else:
                st.info("Sin datos por clase. Usa `--layers confusion` para generarlos.")

        with subtab_trend:
            if run.perclass_csv_path and run.perclass_csv_path.exists():
                pcdf = _load_perclass(str(run.perclass_csv_path))
                classes = sorted(pcdf["class_name"].unique().tolist())
                col_sel, col_met = st.columns([3, 1])
                with col_sel:
                    selected_classes = st.multiselect("Clases (máx 8)", classes, default=classes[:4], max_selections=8)
                with col_met:
                    metric_sel = st.radio("Métrica", ["f1", "precision", "recall"])

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
                        **_base_layout(400, f"{metric_sel.capitalize()} por clase a lo largo de los epochs"),
                        xaxis_title="Epoch",
                    )
                    fig_trend.update_yaxes(range=[0, 1])
                    _show(fig_trend, "tendencia_clases")
            else:
                st.info("Sin CSV de per-class para este run.")

        with subtab_cm:
            if run.confusion_matrix_csv_path and run.confusion_matrix_csv_path.exists():
                cm_df = parse_confusion_matrix_csv(run.confusion_matrix_csv_path)
                epochs_cm = sorted(cm_df["epoch"].unique().tolist())

                col_cm1, col_cm2 = st.columns([3, 1])
                with col_cm1:
                    selected_cm_ep = st.selectbox("Epoch", epochs_cm,
                                                   format_func=lambda e: f"Epoch {e}", key="cm_epoch_sel")
                with col_cm2:
                    cm_mode = st.radio("Modo", ["Normalizada", "Absoluta"], key="cm_mode")

                pivot = get_matrix_for_epoch(cm_df, selected_cm_ep)
                class_order = list(pivot.index)
                z_norm = pivot.reindex(index=class_order, columns=class_order).values
                n_classes = len(class_order)

                if cm_mode == "Absoluta":
                    row_sums = z_norm.sum(axis=1, keepdims=True)
                    z_abs = (z_norm * row_sums).round().astype(int)
                    z_plot = z_abs.tolist()
                    text = [[str(v) if v > 0 else "" for v in row] for row in z_abs]
                    zmin, zmax, cb_title = 0, None, "Muestras"
                else:
                    z_plot = z_norm.tolist()
                    text = [[f"{v:.2f}" if v >= 0.05 else "" for v in row] for row in z_norm]
                    zmin, zmax, cb_title = 0, 1, "P(pred j | verdadero i)"

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
                    hovertemplate="Verdadero: %{y}<br>Predicho: %{x}<br>Valor: %{z:.3f}<extra></extra>",
                    colorbar=dict(title=cb_title),
                ))
                fig_cm.update_layout(
                    title=dict(text=f"Matriz de confusión ({cm_mode.lower()}) — Epoch {selected_cm_ep}",
                               font=dict(size=13)),
                    xaxis=dict(title="Predicho", tickangle=45, tickfont=dict(size=9),
                               tickmode="array", tickvals=list(range(n_classes)), ticktext=class_order),
                    yaxis=dict(title="Verdadero", tickfont=dict(size=9), autorange="reversed",
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
                    "Los bordes de color agrupan las clases por tipo de ecosistema. "
                    "La diagonal = recall por clase.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Sin matriz de confusión. Usa `--layers confusion` para generarla.")

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH
# ═══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    if selected_run is None:
        st.info("Selecciona un run en la barra lateral.")
    elif not run.batch_csv_path:
        st.info(
            "Sin CSV de nivel de batch para este run. "
            "Usa `--layers batch-monitor` para generarlo. "
            "Con `--batch-log-every 1` obtienes un registro por cada batch individual."
        )
    else:
        # ── Sub-pestañas: por epoch | historia global | learning rate ─────────
        subtab_by_ep, subtab_global, subtab_lr = st.tabs(
            ["Por epoch", "Historia global", "Learning rate"]
        )

        # Carga con TTL corto para que el live refresh funcione
        @st.cache_data(ttl=5)
        def _load_batch_live(p: str) -> pd.DataFrame:
            return _load_batch(p)

        bdf = _load_batch_live(str(run.batch_csv_path))
        has_batch_loss = "batch_loss" in bdf.columns and bdf["batch_loss"].notna().any()
        has_lr = "lr" in bdf.columns and bdf["lr"].notna().any()

        # Mapa de métricas disponibles por batch → etiqueta legible
        _BATCH_METRIC_LABELS = {
            "running_loss": "Loss media acumulada",
            "batch_loss": "Loss instantánea por batch",
            "batch_f1": "F1 (macro) por batch",
            "batch_acc": "Accuracy por batch",
            "batch_prec": "Precision (macro) por batch",
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
            st.warning("El CSV de batch está vacío.")
        else:
            epochs_available_b = sorted(bdf["epoch"].unique())
            n_batches_total = int(bdf["n_batches"].iloc[0]) if not bdf.empty else "—"

            # Resumen rápido
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Epochs registrados", len(epochs_available_b))
            bc2.metric("Batches por epoch", n_batches_total)
            bc3.metric("Total registros", len(bdf))
            if has_lr:
                bc4.metric("LR inicial", f"{bdf['lr'].iloc[0]:.2e}")

            # ── Tab: por epoch ────────────────────────────────────────────────
            with subtab_by_ep:
                col_ep, col_met, col_ma = st.columns([2, 2, 2])
                with col_ep:
                    selected_epochs_b = st.multiselect(
                        "Epochs", epochs_available_b,
                        default=list(epochs_available_b[-min(3, len(epochs_available_b)):]),
                    )
                with col_met:
                    batch_metric = st.selectbox(
                        "Métrica", _available_batch_metrics(),
                        format_func=lambda m: _BATCH_METRIC_LABELS.get(m, m),
                    )
                with col_ma:
                    ma_window = st.slider("Media móvil (batches)", 0, 200, 10,
                                          help="0 = desactivado")

                if selected_epochs_b and batch_metric in bdf.columns:
                    fig_b = go.Figure()
                    for i, ep in enumerate(selected_epochs_b):
                        subset = bdf[bdf["epoch"] == ep].copy().sort_values("batch")
                        color = COLORS[i % len(COLORS)]

                        # Línea base (semitransparente)
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

                        # Detección de picos
                        mean_l = subset[batch_metric].mean()
                        std_l = subset[batch_metric].std()
                        if not pd.isna(std_l) and std_l > 0:
                            spikes = subset[subset[batch_metric] > mean_l + 2.5 * std_l]
                            if not spikes.empty:
                                fig_b.add_trace(go.Scatter(
                                    x=spikes["batch"], y=spikes[batch_metric],
                                    name=f"Pico Ep{ep}", mode="markers",
                                    marker=dict(color="red", size=7, symbol="x"),
                                    legendgroup=f"ep{ep}", showlegend=False,
                                ))

                    y_label = _BATCH_METRIC_LABELS.get(batch_metric, batch_metric)
                    fig_b.update_layout(
                        **_base_layout(420, f"{y_label}"),
                        xaxis_title="Batch dentro del epoch",
                        yaxis_title=y_label,
                    )
                    # Las métricas F1/acc/prec van en [0,1]
                    if not _is_loss_metric(batch_metric):
                        fig_b.update_yaxes(range=[0, 1])
                    _show(fig_b, f"batch_{batch_metric}_por_epoch")
                    sel_bdf = bdf[bdf["epoch"].isin(selected_epochs_b)]
                    _dl_csv(sel_bdf, "batch_metrics_sel.csv", "Descargar datos seleccionados")

                    with st.expander("Datos en bruto"):
                        st.dataframe(sel_bdf, use_container_width=True)

            # ── Tab: historia global (eje x = batch global) ───────────────────
            with subtab_global:
                st.markdown(
                    "Vista completa de toda la historia de entrenamiento en un solo eje. "
                    "Las líneas verticales marcan los límites de epoch."
                )
                col_gm, col_gma = st.columns([2, 2])
                with col_gm:
                    global_metric = st.selectbox(
                        "Métrica global", _available_batch_metrics(),
                        format_func=lambda m: _BATCH_METRIC_LABELS.get(m, m),
                        key="global_metric_sel",
                    )
                with col_gma:
                    gma_window = st.slider("Media móvil (batches globales)", 0, 500, 50,
                                           help="0 = desactivado", key="global_ma")

                if global_metric in bdf.columns:
                    all_sorted = bdf.sort_values("global_batch")
                    fig_g = go.Figure()

                    # Serie completa (semitransparente)
                    fig_g.add_trace(go.Scatter(
                        x=all_sorted["global_batch"], y=all_sorted[global_metric],
                        name="Datos", mode="lines",
                        line=dict(color=COLORS[0], width=1), opacity=0.3,
                    ))

                    if gma_window > 1 and len(all_sorted) >= gma_window:
                        gma = all_sorted[global_metric].rolling(gma_window, center=True).mean()
                        fig_g.add_trace(go.Scatter(
                            x=all_sorted["global_batch"], y=gma,
                            name=f"MA{gma_window}", mode="lines",
                            line=dict(color=COLORS[0], width=2.5),
                        ))

                    # Líneas verticales por epoch
                    epoch_boundaries = bdf.groupby("epoch")["global_batch"].max()
                    for ep, gb in epoch_boundaries.items():
                        fig_g.add_vline(
                            x=gb, line_dash="dot", line_color="#94a3b8", line_width=1,
                            annotation_text=f"E{ep}", annotation_position="top",
                            annotation_font_size=9,
                        )

                    y_label_g = _BATCH_METRIC_LABELS.get(global_metric, global_metric)
                    fig_g.update_layout(
                        **_base_layout(420, f"{y_label_g} — historia completa"),
                        xaxis_title="Batch global",
                        yaxis_title=y_label_g,
                    )
                    if not _is_loss_metric(global_metric):
                        fig_g.update_yaxes(range=[0, 1])
                    _show(fig_g, "batch_historia_global")
                    _dl_csv(bdf, "batch_metrics_completo.csv", "Descargar historia completa")

            # ── Tab: learning rate ────────────────────────────────────────────
            with subtab_lr:
                if not has_lr:
                    st.info(
                        "No hay datos de learning rate. "
                        "Requiere `--layers batch-monitor` con la versión actual del BatchMonitorDecorator."
                    )
                else:
                    lr_sorted = bdf.sort_values("global_batch").dropna(subset=["lr"])
                    fig_lr = go.Figure()
                    fig_lr.add_trace(go.Scatter(
                        x=lr_sorted["global_batch"], y=lr_sorted["lr"],
                        name="Learning rate", mode="lines",
                        line=dict(color=COLORS[2], width=2),
                    ))

                    # Marcar boundaries de epoch
                    epoch_boundaries_lr = bdf.groupby("epoch")["global_batch"].max()
                    for ep, gb in epoch_boundaries_lr.items():
                        fig_lr.add_vline(
                            x=gb, line_dash="dot", line_color="#94a3b8", line_width=1,
                            annotation_text=f"E{ep}", annotation_position="top",
                            annotation_font_size=9,
                        )

                    fig_lr.update_layout(
                        **_base_layout(380, "Evolución del learning rate"),
                        xaxis_title="Batch global",
                        yaxis_title="Learning rate",
                    )
                    # Escala log si el rango es grande
                    lr_range = lr_sorted["lr"].max() / (lr_sorted["lr"].min() + 1e-12)
                    if lr_range > 100:
                        fig_lr.update_yaxes(type="log")
                    _show(fig_lr, "learning_rate")

                    # Stats de LR
                    lr_col1, lr_col2, lr_col3 = st.columns(3)
                    lr_col1.metric("LR inicial", f"{lr_sorted['lr'].iloc[0]:.2e}")
                    lr_col2.metric("LR final", f"{lr_sorted['lr'].iloc[-1]:.2e}")
                    lr_col3.metric("LR mínimo", f"{lr_sorted['lr'].min():.2e}")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARAR
# ═══════════════════════════════════════════════════════════════════════════════

with tab_comparar:
    if not runs:
        st.info("No hay runs disponibles.")
    else:
        all_run_labels = {r.label: r for r in runs}
        all_labels_list = list(all_run_labels.keys())

        selected_compare = st.multiselect(
            "Selecciona runs a comparar (máx 4)", all_labels_list,
            default=all_labels_list[:min(2, len(all_labels_list))],
            max_selections=4,
        )

        if len(selected_compare) < 2:
            st.info("Selecciona al menos 2 runs.")
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
                    "Mejor Val F1": f"{best_f1_c:.4f}" if not pd.isna(best_f1_c) else "—",
                    "Mejor epoch": int(best_ep_c_v) if best_ep_c_v is not None else "—",
                    "F1 final": f"{_last.iloc[-1]:.4f}" if not _last.empty else "—",
                    "Epochs": len(cdf),
                    "Duración": _dur_str(total_s_c) if not pd.isna(total_s_c) else "—",
                    "Entorno": r.env, "Trace": r.trace_mode,
                })

            sum_df = pd.DataFrame(summary_rows).set_index("Run")
            st.dataframe(sum_df, use_container_width=True)
            _dl_csv(sum_df.reset_index(), "comparativa_runs.csv", "Descargar comparativa")
            st.markdown("---")

            st.markdown("#### Radar de métricas en el mejor epoch")
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
                title=dict(text="Métricas en el mejor epoch de Val F1", font=dict(size=13)),
            )
            _show(radar_fig, "radar_comparativa")
            st.markdown("---")

            metrics_to_compare = st.multiselect(
                "Métricas a superponer",
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
    subtab_report, subtab_ddp_opt, subtab_prediction, subtab_compare_feas, subtab_run_feas = st.tabs(
        ["Informe", "Análisis DDP", "Predicción F1", "Comparar vs training", "Ejecutar análisis"]
    )

    # Carga común del informe seleccionado
    feasibility_csvs = _get_feasibility_csvs()
    if feasibility_csvs:
        csv_labels_feas = {str(p): f"{p.parent.name}/{p.name}" for p in feasibility_csvs}
        selected_feas_path = st.sidebar.selectbox(
            "Informe viabilidad", list(csv_labels_feas.keys()),
            format_func=lambda p: csv_labels_feas[p], key="feas_sidebar_sel",
        )
        meta, bdf_feas = parse_feasibility_csv(Path(selected_feas_path))
    else:
        meta, bdf_feas = {}, pd.DataFrame()

    # ── Informe ───────────────────────────────────────────────────────────────
    with subtab_report:
        if not feasibility_csvs:
            st.info("No se encontraron CSVs de viabilidad. Ejecuta el análisis desde la sub-pestaña 'Ejecutar análisis'.")
        else:
            # ── Perfil del sistema ────────────────────────────────────────────
            st.markdown("### Perfil del sistema")
            hw_col1, hw_col2, hw_col3, hw_col4 = st.columns(4)
            hw_col1.metric("Modelo", meta.get("model_name", "—"))
            hw_col2.metric("Parámetros (M)", meta.get("total_params_M", "—"))
            hw_col3.metric("GPU", meta.get("hardware_name", "—"))
            hw_col4.metric("VRAM total (GB)", meta.get("total_vram_gb", "—"))

            # CPU si disponible
            cpu = meta.get("cpu", {})
            if cpu:
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Núcleos lógicos", cpu.get("logical_cores", "—"))
                cc2.metric("Núcleos físicos", cpu.get("physical_cores", "—"))
                cc3.metric("RAM total (GB)", cpu.get("ram_total_gb", "—"))
                cc4.metric("RAM libre (GB)", cpu.get("ram_free_gb", "—"))

            # Disco si disponible
            disk = meta.get("disk", {})
            ds_profile = meta.get("dataset", {})
            if disk or ds_profile:
                st.markdown("### I/O del dataset")
                di_cols = st.columns(4)
                if disk:
                    di_cols[0].metric("Tipo de disco", disk.get("type", "—"))
                    di_cols[1].metric("NFS", "Sí" if disk.get("is_nfs") == "yes" else "No")
                    if disk.get("read_mb_per_s", "0") != "0":
                        di_cols[2].metric("Velocidad lectura", f"{disk.get('read_mb_per_s', '—')} MB/s")
                        di_cols[3].metric("Patches/s", f"{disk.get('files_per_second', '—')}")
                if ds_profile:
                    io_ratio = float(ds_profile.get("io_bottleneck_ratio", 0) or 0)
                    st.metric("Ratio I/O vs cómputo", f"{io_ratio:.2f}",
                               delta="I/O-bound" if io_ratio > 1.2 else "Compute-bound",
                               delta_color="inverse" if io_ratio > 1.2 else "normal")
                    if io_ratio > 1.2:
                        st.warning("Cuello de botella en I/O: el data loading es más lento que el cómputo GPU. Más GPUs no mejorarán el throughput sin un disco más rápido.")
                    else:
                        st.success("Compute-bound: la GPU es el cuello de botella. Añadir GPUs (DDP) acelerará el training linealmente.")

            # Memoria por batch size
            mem_keys = ["weight_mb", "gradient_mb", "optimizer_mb", "activation_mb_per_image", "total_static_mb"]
            if any(k in meta for k in mem_keys):
                st.markdown("### Memoria del modelo")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Pesos (MB)", meta.get("weight_mb", "—"))
                m2.metric("Gradientes (MB)", meta.get("gradient_mb", "—"))
                m3.metric("AdamW state (MB)", meta.get("optimizer_mb", "—"))
                m4.metric("Activaciones/img (MB)", meta.get("activation_mb_per_image", "—"))
                m5.metric("Total estático (MB)", meta.get("total_static_mb", "—"))

                # VRAM visual
                total_vram = meta.get("total_vram_gb")
                free_vram = meta.get("free_vram_gb")
                if total_vram and free_vram:
                    fig_vr = go.Figure(go.Bar(
                        x=["Libre", "Usada"],
                        y=[float(free_vram), float(total_vram) - float(free_vram)],
                        marker_color=[COLORS[2], COLORS[3]], opacity=0.85,
                    ))
                    fig_vr.update_layout(**_base_layout(180, "Distribución VRAM"), yaxis_title="GB")
                    _show(fig_vr, "vram_dist")

            # Benchmark
            if not bdf_feas.empty:
                st.markdown("### Benchmark de throughput")
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
                    fig_tp.update_layout(**_base_layout(300, "Throughput (imgs/s) por batch size"),
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
                                annotation_text=f"VRAM libre: {meta['free_vram_gb']} GB",
                                annotation_position="top left",
                            )
                        fig_vram_f.update_layout(**_base_layout(260, "VRAM pico por batch size"),
                                                  barmode="group", xaxis_title="Batch size", yaxis_title="GB")
                        _show(fig_vram_f, "vram_pico")

                st.dataframe(bdf_feas, use_container_width=True, height=220)
                _dl_csv(bdf_feas, "feasibility_benchmark.csv", "Descargar benchmark")

                # Estimaciones de tiempo
                est_cols = [c for c in bdf_feas.columns if c.startswith("est_")]
                if est_cols:
                    st.markdown("### Estimaciones de tiempo")
                    orig_ep_col = next(
                        (c for c in bdf_feas.columns if c.startswith("est_total_h_") and c.endswith("ep")), None
                    )
                    orig_n = None
                    if orig_ep_col:
                        try:
                            orig_n = int(orig_ep_col.split("est_total_h_")[1].replace("ep", ""))
                        except ValueError:
                            pass
                    recalc_n = st.number_input("Epochs para estimación total", min_value=1, value=orig_n or 30)
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
                    _dl_csv(est_df, "estimaciones_tiempo.csv", "Descargar estimaciones")

    # ── Análisis DDP ──────────────────────────────────────────────────────────
    with subtab_ddp_opt:
        if not feasibility_csvs:
            st.info("Ejecuta primero el análisis de viabilidad.")
        else:
            st.markdown("## Análisis DDP — Distribución óptima de recursos")
            st.caption(
                "Compara configuraciones de 1 a 8 GPUs mostrando batch size, workers recomendados, "
                "speedup esperado, eficiencia de escalado y cuello de botella identificado."
            )
            ddp_df = parse_ddp_scenarios(meta)

            if ddp_df.empty:
                st.info(
                    "No hay datos DDP en este informe. "
                    "Regenera el análisis con la versión actual de check_feasibility.py."
                )
            else:
                # ── Tabla de escenarios ───────────────────────────────────────
                st.markdown("### Tabla de escenarios")

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
                _dl_csv(ddp_df, "ddp_scenarios.csv", "Descargar escenarios DDP")

                # ── Rectángulos de distribución de carga ─────────────────────
                st.markdown("### Distribución de carga por GPU")
                st.caption(
                    "Cada barra muestra la proporción del tiempo de batch: "
                    "cómputo (verde), I/O de datos (naranja), sincronización de gradientes (rojo)."
                )

                # Calcular proporciones por GPU
                if {"speedup", "sync_overhead_pct", "n_gpus"}.issubset(ddp_df.columns):
                    viable_ddp = ddp_df[pd.to_numeric(ddp_df["n_gpus"], errors="coerce") > 0].copy()
                    for col in ["sync_overhead_pct", "speedup", "n_gpus"]:
                        viable_ddp[col] = pd.to_numeric(viable_ddp[col], errors="coerce")

                    # Estimar I/O overhead desde el ratio si está disponible
                    io_ratio = float(meta.get("dataset", {}).get("io_bottleneck_ratio", 0) or 0)
                    io_pct_est = min(io_ratio * 30, 50)  # Estimación: si ratio=1, I/O ≈ 30% del tiempo

                    fig_rect = go.Figure()
                    labels = [f"{int(row['n_gpus'])} GPU(s)" for _, row in viable_ddp.iterrows()]
                    sync_pcts = viable_ddp["sync_overhead_pct"].fillna(0).tolist()
                    compute_pcts = [max(0, 100 - s - io_pct_est) for s in sync_pcts]
                    io_pcts = [io_pct_est] * len(labels)

                    fig_rect.add_trace(go.Bar(
                        name="Cómputo GPU", x=labels, y=compute_pcts,
                        marker_color=COLORS[2], opacity=0.85,
                    ))
                    fig_rect.add_trace(go.Bar(
                        name="I/O datos", x=labels, y=io_pcts,
                        marker_color=COLORS[1], opacity=0.85,
                    ))
                    fig_rect.add_trace(go.Bar(
                        name="Sync gradientes", x=labels, y=sync_pcts,
                        marker_color=COLORS[3], opacity=0.85,
                    ))
                    fig_rect.update_layout(
                        **_base_layout(360, "Desglose de tiempo por batch (%) — estimación"),
                        barmode="stack",
                        xaxis_title="Configuración DDP",
                        yaxis_title="Porcentaje del tiempo de batch",
                    )
                    fig_rect.update_yaxes(range=[0, 100])
                    _show(fig_rect, "ddp_carga_distribucion")

                    # ── Speedup vs teórico ────────────────────────────────────
                    st.markdown("### Speedup: real vs teórico")
                    if "speedup" in viable_ddp.columns:
                        n_gpus_vals = viable_ddp["n_gpus"].tolist()
                        speedup_vals = viable_ddp["speedup"].tolist()
                        theoretical = n_gpus_vals  # speedup teórico lineal

                        fig_su = go.Figure()
                        fig_su.add_trace(go.Scatter(
                            x=n_gpus_vals, y=theoretical,
                            name="Teórico (100% eficiencia)",
                            mode="lines+markers", line=dict(color=COLORS[4], width=2, dash="dash"),
                        ))
                        fig_su.add_trace(go.Scatter(
                            x=n_gpus_vals, y=speedup_vals,
                            name="Speedup real estimado",
                            mode="lines+markers",
                            line=dict(color=COLORS[2], width=3),
                            marker=dict(size=10),
                        ))
                        fig_su.update_layout(
                            **_base_layout(320, "Speedup real vs teórico"),
                            xaxis_title="Número de GPUs",
                            yaxis_title="Speedup",
                        )
                        fig_su.update_xaxes(tickvals=n_gpus_vals)
                        _show(fig_su, "ddp_speedup")

                    # ── Tiempo total estimado por configuración ───────────────
                    if "time_total_h" in viable_ddp.columns:
                        st.markdown("### Tiempo total estimado por configuración")
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
                            **_base_layout(280, "Tiempo total de entrenamiento (h)"),
                            xaxis_title="Configuración DDP",
                            yaxis_title="Horas",
                        )
                        _show(fig_tt, "ddp_tiempo_total")

    # ── Predicción de rendimiento F1 ──────────────────────────────────────────
    with subtab_prediction:
        if not feasibility_csvs:
            st.info("Ejecuta primero el análisis de viabilidad.")
        else:
            st.markdown("## Predicción empírica de rendimiento")
            pred = meta.get("prediction", {})
            curve_val = meta.get("curve_val_f1", [])
            curve_train = meta.get("curve_train_f1", [])
            curve_epochs = meta.get("curve_epochs", [])

            if not pred:
                st.info(
                    "No hay datos de predicción en este informe. "
                    "Regenera con la versión actual de check_feasibility.py."
                )
            else:
                # ── Métricas clave de predicción ──────────────────────────────
                pred_best_f1 = float(pred.get("predicted_best_f1", 0) or 0)
                pred_best_ep = int(float(pred.get("predicted_best_epoch", 0) or 0))
                pred_stop_ep = int(float(pred.get("predicted_early_stop_epoch", 0) or 0))
                confidence = pred.get("confidence", "—")

                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Val F1 esperado", f"{pred_best_f1:.3f}")
                pc2.metric("Mejor epoch estimado", pred_best_ep)
                pc3.metric("Early stop estimado", pred_stop_ep)
                pc4.metric("Confianza", confidence)

                # ── Curva F1 predicha ─────────────────────────────────────────
                if curve_val and curve_epochs:
                    st.markdown("### Curva F1 estimada")
                    st.caption(
                        "Predicción basada en datos históricos de entrenamientos reales en BigEarthNet-S2. "
                        "La banda de incertidumbre refleja la variabilidad observada (±0.008 F1 entre runs)."
                    )

                    fig_pred = go.Figure()

                    # Banda de incertidumbre
                    uncertainty = 0.015
                    fig_pred.add_trace(go.Scatter(
                        x=curve_epochs + curve_epochs[::-1],
                        y=[v + uncertainty for v in curve_val] + [v - uncertainty for v in curve_val[::-1]],
                        fill="toself", fillcolor="rgba(37,99,235,0.1)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="Incertidumbre (±0.015 F1)",
                        showlegend=True,
                    ))

                    # Val F1 predicho
                    fig_pred.add_trace(go.Scatter(
                        x=curve_epochs, y=curve_val,
                        name="Val F1 estimado",
                        mode="lines", line=dict(color=COLORS[0], width=3),
                    ))

                    # Train F1 predicho
                    if curve_train:
                        fig_pred.add_trace(go.Scatter(
                            x=curve_epochs, y=curve_train,
                            name="Train F1 estimado",
                            mode="lines", line=dict(color=COLORS[0], width=2, dash="dot"),
                            opacity=0.6,
                        ))

                    # Marcar mejor epoch
                    if pred_best_ep <= max(curve_epochs):
                        best_val = curve_val[pred_best_ep - 1] if pred_best_ep <= len(curve_val) else pred_best_f1
                        fig_pred.add_trace(go.Scatter(
                            x=[pred_best_ep], y=[best_val],
                            name=f"Mejor epoch ({pred_best_ep})",
                            mode="markers", marker=dict(color="gold", size=14, symbol="star"),
                        ))

                    # Marcar early stop
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
                                    name="Val F1 real",
                                    mode="lines+markers",
                                    line=dict(color=COLORS[1], width=2.5),
                                    marker=dict(size=5),
                                ))
                        except Exception:
                            pass

                    fig_pred.update_layout(
                        **_base_layout(420, "Curva F1 de validación — predicción vs real"),
                        xaxis_title="Epoch",
                        yaxis_title="Val F1 (macro)",
                    )
                    fig_pred.update_yaxes(range=[0.0, 1.0])
                    _show(fig_pred, "prediccion_f1")

                    if selected_run is not None:
                        st.caption(
                            "Línea amarilla = predicción empírica | "
                            "Naranja = datos reales del run seleccionado | "
                            "Estrella dorada = mejor epoch estimado"
                        )
                    else:
                        st.caption(
                            "Selecciona un run en la barra lateral para superponer los resultados reales."
                        )

                # Datos de predicción como tabla descargable
                if curve_val and curve_epochs:
                    import pandas as pd
                    pred_curve_df = pd.DataFrame({
                        "epoch": curve_epochs,
                        "val_f1_pred": curve_val,
                        "train_f1_pred": curve_train if curve_train else [None] * len(curve_epochs),
                        "val_f1_upper": [v + 0.015 for v in curve_val],
                        "val_f1_lower": [v - 0.015 for v in curve_val],
                    })
                    _dl_csv(pred_curve_df, "prediccion_curva_f1.csv", "Descargar curva predicha")

    # ── Comparar vs training ──────────────────────────────────────────────────
    with subtab_compare_feas:
        st.markdown("### Estimaciones de viabilidad vs resultados reales de training")
        feasibility_csvs_cmp = _get_feasibility_csvs()
        all_runs_cmp = _get_runs()

        if not feasibility_csvs_cmp:
            st.info("No se encontraron CSVs de viabilidad.")
        elif not all_runs_cmp:
            st.info("No se encontraron runs de training.")
        else:
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                csv_labels_cmp = {str(p): f"{p.parent.name}/{p.name}" for p in feasibility_csvs_cmp}
                sel_feas_cmp = st.selectbox("Informe de viabilidad", list(csv_labels_cmp.keys()),
                                             format_func=lambda p: csv_labels_cmp[p], key="cmp_feas_sel")
                meta_cmp, feas_df_cmp = parse_feasibility_csv(Path(sel_feas_cmp))
                model_feas = meta_cmp.get("model_name", "")

                batch_sizes_available = []
                if not feas_df_cmp.empty and "batch_size" in feas_df_cmp.columns:
                    batch_sizes_available = sorted(
                        feas_df_cmp["batch_size"].dropna().astype(int).unique().tolist()
                    )
                sel_bs = (
                    st.selectbox("Batch size", batch_sizes_available, key="cmp_bs_sel")
                    if batch_sizes_available else None
                )
                trace_modes_available = []
                if not feas_df_cmp.empty and "trace_mode" in feas_df_cmp.columns:
                    trace_modes_available = sorted(feas_df_cmp["trace_mode"].unique().tolist())
                sel_trace = st.selectbox("Trace mode", trace_modes_available or ["simple"], key="cmp_trace_sel")
                nfs_factor_cmp = float(meta_cmp.get("nfs_factor", 1.0) or 1.0)
                st.caption(f"Modelo: **{model_feas or '—'}** | Factor NFS: {nfs_factor_cmp:.2f}")

            with cmp_col2:
                run_labels_cmp = {r.label: r for r in all_runs_cmp}
                matching = [lbl for lbl, r in run_labels_cmp.items()
                            if model_feas and r.model and model_feas in r.model]
                default_run = matching[0] if matching else list(run_labels_cmp.keys())[0]
                sel_run_cmp = st.selectbox(
                    "Run de training", list(run_labels_cmp.keys()),
                    index=list(run_labels_cmp.keys()).index(default_run), key="cmp_run_sel",
                )
                run_cmp = run_labels_cmp[sel_run_cmp]
                actual_df_cmp = _load_df(
                    str(run_cmp.log_path),
                    str(run_cmp.epoch_csv_path) if run_cmp.epoch_csv_path else None,
                )

            if sel_bs is not None and not actual_df_cmp.empty:
                comparison = build_comparison(
                    meta=meta_cmp, feas_df=feas_df_cmp, actual_df=actual_df_cmp,
                    batch_size=int(sel_bs), trace_mode=sel_trace, nfs_factor=nfs_factor_cmp,
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

                    err_col = next((c for c in cmp_table.columns if c == "Error %"), None)
                    styled_cmp = cmp_table.style
                    if err_col:
                        styled_cmp = styled_cmp.map(_color_error, subset=[err_col])
                    st.dataframe(styled_cmp, use_container_width=True, hide_index=True)
                    _dl_csv(cmp_table, "comparativa_viabilidad.csv", "Descargar comparativa")
                    st.caption("Verde = error ≤ 10% | Amarillo = 10–30% | Rojo = > 30%.")
                else:
                    st.warning(f"Sin fila coincidente para batch_size={sel_bs}, trace_mode={sel_trace}.")

    # ── Ejecutar análisis ─────────────────────────────────────────────────────
    with subtab_run_feas:
        st.subheader("Ejecutar análisis de viabilidad")
        configs_available = _get_configs()
        model_options_f = [
            "vit_base_patch16_224", "vit_tiny_patch16_224",
            "vit_small_patch16_224", "resnet50", "efficientnet_b0",
        ]
        with st.form("feasibility_form"):
            fa1, fa2 = st.columns(2)
            with fa1:
                feas_model = st.selectbox("Modelo", model_options_f)
                feas_batches = st.multiselect("Batch sizes", [16, 32, 64, 128], default=[32, 64])
                feas_epochs = st.number_input("Epochs para estimación", min_value=1, value=30)
                feas_dataset_path = st.text_input(
                    "Ruta al dataset (opcional — para medir I/O real)",
                    placeholder="/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2",
                )
            with fa2:
                feas_traces = st.multiselect("Trace modes", ["off", "simple", "deep"],
                                              default=["off", "simple"])
                feas_nfs = st.slider("Factor NFS", 1.0, 2.0, 1.0, 0.05,
                                     help="Corrección para latencia NFS (Verode: ~1.3)")
                feas_config = st.selectbox(
                    "Config YAML (opcional)",
                    ["(ninguno)"] + (configs_available if configs_available else []),
                )
                feas_no_disk = st.checkbox("Omitir medición de I/O (más rápido)", value=False)
            submitted_feas = st.form_submit_button("Ejecutar")

        if submitted_feas:
            if not feas_batches:
                st.error("Selecciona al menos un batch size.")
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
                if feas_config != "(ninguno)":
                    parts.append(f"--config configs/{feas_config}")
                if feas_dataset_path.strip():
                    parts.append(f'--dataset-path "{feas_dataset_path.strip()}"')
                if feas_no_disk:
                    parts.append("--no-disk-profile")
                cmd = " ".join(parts)
                st.code(cmd, language="bash")
                out_ph = st.empty()
                with st.spinner("Ejecutando análisis completo…"):
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(ROOT))
                if result.returncode == 0:
                    st.success("Análisis completado.")
                    out_ph.code(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
                    _get_feasibility_csvs.clear()
                else:
                    st.error("Error durante el análisis:")
                    out_ph.code(result.stderr[-2000:])

# ═══════════════════════════════════════════════════════════════════════════════
# TIEMPO
# ═══════════════════════════════════════════════════════════════════════════════

with tab_tiempo:
    if selected_run is None:
        st.info("Selecciona un run en la barra lateral.")
    else:
        df_time = _load_df(str(run.log_path), str(run.epoch_csv_path) if run.epoch_csv_path else None)

        if "epoch_time" not in df_time.columns or df_time["epoch_time"].isna().all():
            st.info("Sin datos de tiempo por epoch. Usa `--trace simple` para generarlos.")
        else:
            et = df_time[["epoch", "epoch_time"]].dropna()
            total_s_t = et["epoch_time"].sum()
            avg_s_t = et["epoch_time"].mean()

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Total", _dur_str(total_s_t))
            t2.metric("Promedio/epoch", f"{avg_s_t/60:.1f} min")
            t3.metric("Mínimo/epoch", f"{et['epoch_time'].min()/60:.1f} min")
            t4.metric("Máximo/epoch", f"{et['epoch_time'].max()/60:.1f} min")

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
                                              name="Tendencia", mode="lines",
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
            if feasibility_csvs_t:
                try:
                    _, bdf_t = parse_feasibility_csv(feasibility_csvs_t[0])
                    viable_t = bdf_t[bdf_t["oom"] == "no"].copy()
                    tp_col_t = _throughput_col(viable_t)
                    per_ep_col = next((c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                                       if c in viable_t.columns), None)
                    _idx_t = (viable_t[tp_col_t].idxmax()
                               if (tp_col_t and per_ep_col and not viable_t.empty) else None)
                    if _idx_t is not None and not pd.isna(_idx_t):
                        est_min = float(viable_t.loc[_idx_t, per_ep_col])
                        fig_time.add_hline(y=est_min, line_dash="dash", line_color=COLORS[1],
                                           annotation_text=f"Estimación viabilidad: {est_min:.0f} min/epoch",
                                           annotation_position="top right")
                except Exception:
                    pass

            fig_time.update_layout(**_base_layout(380, "Tiempo por epoch"),
                                   xaxis_title="Epoch", yaxis_title="Minutos")
            _show(fig_time, "tiempo_por_epoch")
            _dl_csv(et.assign(epoch_time_min=et["epoch_time"] / 60),
                    "tiempo_por_epoch.csv", "Descargar datos de tiempo")

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
                        st.markdown("**Estimado vs Real**")
                        ce1, ce2, ce3 = st.columns(3)
                        ce1.metric("Estimado (min/epoch)", f"{est_val:.1f}")
                        ce2.metric("Real promedio (min/epoch)", f"{real_val:.1f}")
                        ce3.metric("Error relativo", f"{err_pct:+.1f}%")
                except Exception:
                    pass

# ═══════════════════════════════════════════════════════════════════════════════
# INFORMACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

with tab_info:
    if selected_run is None:
        st.info("Selecciona un run en la barra lateral.")
    else:
        df_info = _load_df(str(run.log_path), str(run.epoch_csv_path) if run.epoch_csv_path else None)
        n_ep_i = len(df_info)
        best_f1_i = _safe_max(df_info["val_f1"]) if "val_f1" in df_info.columns else float("nan")
        best_ep_i_v = _safe_val_at_best(df_info, "val_f1", "epoch")

        col_m, col_f = st.columns(2)

        with col_m:
            st.subheader("Metadatos del run")
            rows_i = {
                "Log": run.log_path.name,
                "Entorno": run.env,
                "Trace mode": run.trace_mode,
                "Epochs": n_ep_i,
                "Mejor Val F1": f"{best_f1_i:.4f}" if not pd.isna(best_f1_i) else "—",
                "Mejor epoch": int(best_ep_i_v) if best_ep_i_v is not None else "—",
            }
            if "epoch_time" in df_info.columns and df_info["epoch_time"].notna().any():
                total_si = df_info["epoch_time"].sum()
                rows_i["Tiempo total"] = _dur_str(total_si)
                rows_i["Promedio/epoch"] = f"{df_info['epoch_time'].mean()/60:.1f} min"
            for k, v in rows_i.items():
                st.markdown(f"**{k}:** {v}")

        with col_f:
            st.subheader("Ficheros asociados")
            for label_f, path_f in [
                ("Batch CSV", run.batch_csv_path),
                ("Per-class CSV", run.perclass_csv_path),
                ("Epoch CSV", run.epoch_csv_path),
                ("Confusion matrix CSV", run.confusion_matrix_csv_path),
            ]:
                st.markdown(f"- **{label_f}:** `{path_f.name if path_f else '—'}`")

        st.markdown("---")

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
            st.caption("No se pudo determinar la config de este run.")

        st.subheader("Detección de anomalías")
        anomalies = _detect_anomalies(run.log_path)
        if anomalies:
            st.warning(f"{len(anomalies)} línea(s) con anomalías detectadas.")
            with st.expander("Ver anomalías"):
                for line in anomalies:
                    st.text(line)
        else:
            st.success("Sin anomalías detectadas en el log.")

        st.subheader("Log")
        search_term = st.text_input("Filtrar líneas del log", "")
        try:
            all_lines = run.log_path.read_text(errors="replace").splitlines()
            if search_term:
                disp_lines = [ln for ln in all_lines if search_term.lower() in ln.lower()]
                st.caption(f"{len(disp_lines)} / {len(all_lines)} líneas")
            else:
                disp_lines = all_lines
                st.caption(f"{len(all_lines)} líneas en total")
            st.code("\n".join(disp_lines[-400:]), language="text")
        except Exception as exc:
            st.error(str(exc))

# ═══════════════════════════════════════════════════════════════════════════════
# LANZADOR
# ═══════════════════════════════════════════════════════════════════════════════

with tab_lanzador:
    subtab_single, subtab_ddp_l = st.tabs(["Single GPU", "DDP (multi-GPU)"])

    configs_l = _get_configs()
    model_opts_l = [
        "vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224",
        "resnet50", "efficientnet_b0", "deit_tiny_patch16_224",
    ]

    with subtab_single:
        st.subheader("Entrenamiento Single GPU")
        with st.form("launcher_single_form"):
            la1, la2 = st.columns(2)
            with la1:
                l_model = st.selectbox("Modelo", model_opts_l)
                l_config = st.selectbox("Config YAML", configs_l if configs_l else ["(ninguno)"])
                l_epochs = st.number_input("Override epochs", min_value=0, value=0,
                                           help="0 = usar valor del config")
                l_batch = st.number_input("Override batch size", min_value=0, value=0,
                                          help="0 = usar valor del config")
            with la2:
                l_trace = st.selectbox("Trace mode", ["simple", "off", "deep"])
                l_layers = st.multiselect("Layers", ["plot", "hooks", "confusion", "batch-monitor"],
                                          default=["confusion"])
                l_fn = st.multiselect("Fn decorators", ["timing", "energy"])
                l_inspect = st.multiselect("Inspect features",
                                           ["model-summary", "grad-monitor", "anomalies", "batch-table"])
            launched_single = st.form_submit_button("Lanzar")

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
                st.success("Entrenamiento completado.")
                _get_runs.clear()
            else:
                st.error(f"Proceso terminó con código {rc_l}.")

    with subtab_ddp_l:
        st.subheader("Entrenamiento DDP")
        with st.form("launcher_ddp_form"):
            dd1, dd2 = st.columns(2)
            with dd1:
                d_nproc = st.number_input("GPUs (--nproc_per_node)", min_value=1, max_value=8, value=2)
                d_model = st.selectbox("Modelo", model_opts_l, key="ddp_model")
                d_config = st.selectbox("Config YAML", configs_l if configs_l else ["(ninguno)"],
                                         key="ddp_config")
                d_epochs = st.number_input("Override epochs", min_value=0, value=0, key="ddp_ep")
            with dd2:
                d_trace = st.selectbox("Trace mode", ["simple", "off", "deep"], key="ddp_trace")
                d_layers = st.multiselect("Layers", ["plot", "confusion", "batch-monitor"],
                                          default=["confusion"], key="ddp_layers")
                d_fn = st.multiselect("Fn decorators", ["timing", "energy"], key="ddp_fn")
            launched_ddp = st.form_submit_button("Lanzar")

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
                st.success("Entrenamiento DDP completado.")
                _get_runs.clear()
            else:
                st.error(f"Proceso terminó con código {rc_d}.")

# ═══════════════════════════════════════════════════════════════════════════════
# EN VIVO
# ═══════════════════════════════════════════════════════════════════════════════

with tab_envivo:
    st.subheader("Monitor en vivo")

    now_ts = time.time()
    recent_runs = [
        r for r in runs
        if r.log_path.exists() and (now_ts - r.log_path.stat().st_mtime) < 1800
    ]

    if not recent_runs:
        st.info(
            "No hay runs activos (ningún log modificado en los últimos 30 min). "
            "Lanza un entrenamiento desde la pestaña Lanzador."
        )
    else:
        live_labels = {r.label: r for r in recent_runs}
        live_sel = st.selectbox("Run activo", list(live_labels.keys()), key="live_run_sel")
        live_run = live_labels[live_sel]

        @st.fragment(run_every=refresh_interval)
        def _live_panel(run: RunInfo):
            _load_df.clear()

            gpu = _gpu_usage()
            if gpu:
                g1, g2, g3, g4 = st.columns(4)
                g1.metric("GPU", gpu["name"])
                g2.metric("VRAM", f"{gpu['mem_used_mb']/1024:.1f} / {gpu['mem_total_mb']/1024:.1f} GB")
                g3.metric("Utilización", f"{gpu['util_pct']}%")
                g4.metric("Temperatura", f"{gpu['temp_c']} °C")
            else:
                st.caption("Info GPU no disponible (nvidia-smi no encontrado).")

            progress = _parse_log_progress(run.log_path)
            if progress["epochs"] > 0:
                pct = progress["epoch"] / progress["epochs"]
                st.progress(pct, text=f"Epoch {progress['epoch']} / {progress['epochs']}")

            if progress["last_val_f1"] is not None:
                m1, m2 = st.columns(2)
                m1.metric("Último Val F1", f"{progress['last_val_f1']:.4f}")
                if progress["last_val_loss"] is not None:
                    m2.metric("Último Val Loss", f"{progress['last_val_loss']:.4f}")

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
                    fig_live.update_layout(**_base_layout(280, "Métricas"), xaxis_title="Epoch")
                    _show(fig_live, "live_metrics")

            st.subheader("Cola del log")
            st.code(_read_log_tail(run.log_path, n=40), language="text")

        _live_panel(live_run)
