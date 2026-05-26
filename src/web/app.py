"""Streamlit web dashboard — Training Dashboard v3."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.web.batch_parser import parse_batch_csv
from src.web.confusion_matrix_parser import get_matrix_for_epoch, parse_confusion_matrix_csv
from src.web.feasibility_parser import parse_feasibility_csv
from src.web.log_parser import parse_log
from src.web.perclass_parser import parse_perclass_csv
from src.web.run_registry import RunInfo, discover_feasibility_csvs, discover_runs

ROOT = Path(__file__).resolve().parents[2]
COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860", "#da8bc3", "#8c8c8c"]

st.set_page_config(
    page_title="Training Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# ── Utility functions ──────────────────────────────────────────────────────────


def _metric_fig(
    df: pd.DataFrame,
    col_train: str,
    col_val: str,
    title: str,
    y_label: str,
    color_train: str = COLORS[0],
    color_val: str = COLORS[1],
    extra_traces: list | None = None,
    height: int = 350,
) -> go.Figure:
    fig = go.Figure()
    if col_train in df.columns and df[col_train].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_train], name="Train",
            mode="lines+markers", line=dict(color=color_train, width=2),
            marker=dict(size=5),
        ))
    if col_val in df.columns and df[col_val].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_val], name="Val",
            mode="lines+markers", line=dict(color=color_val, width=2),
            marker=dict(size=5),
        ))
    for tr in (extra_traces or []):
        fig.add_trace(tr)
    fig.update_layout(
        title=title, xaxis_title="Epoch", yaxis_title=y_label,
        height=height, margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _overlay_fig(
    dfs: list[tuple[str, pd.DataFrame]],
    col: str,
    title: str,
    y_label: str,
    height: int = 380,
) -> go.Figure:
    fig = go.Figure()
    for i, (label, df) in enumerate(dfs):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df[col],
                name=label[:30], mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                marker=dict(size=5),
            ))
    fig.update_layout(
        title=title, xaxis_title="Epoch", yaxis_title=y_label,
        height=height, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _gpu_usage() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0:
            return None
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        if len(parts) < 5:
            return None
        return {
            "name": parts[0],
            "mem_used_mb": int(parts[1]),
            "mem_total_mb": int(parts[2]),
            "util_pct": int(parts[3]),
            "temp_c": int(parts[4]),
        }
    except Exception:
        return None


def _parse_log_progress(log_path: Path) -> dict:
    result = {"epoch": 0, "epochs": 0, "last_val_f1": None, "last_val_loss": None}
    try:
        text = log_path.read_text(errors="replace")
        lines = text.splitlines()
        for line in reversed(lines):
            if "Epoch" in line and "/" in line:
                import re
                m = re.search(r"Epoch\s+(\d+)/(\d+)", line)
                if m:
                    result["epoch"] = int(m.group(1))
                    result["epochs"] = int(m.group(2))
                    break
        for line in reversed(lines):
            if "val_f1" in line or "val=0." in line:
                import re
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


def _detect_anomalies_in_log(log_path: Path) -> list[str]:
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


def _launch_process(cmd: str, placeholder, cwd: Path = ROOT) -> int:
    placeholder.code("", language="text")
    output_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(cwd),
        )
        for raw in proc.stdout:  # type: ignore[union-attr]
            output_lines.append(raw.rstrip())
            placeholder.code("\n".join(output_lines[-120:]), language="text")
        proc.wait()
        return proc.returncode
    except Exception as exc:
        placeholder.error(str(exc))
        return -1


# ── Sidebar ───────────────────────────────────────────────────────────────────

runs = _get_runs()

with st.sidebar:
    st.title("📈 Training Dashboard")
    st.markdown("---")

    if not runs:
        st.warning("No runs found in logs/.")
        selected_run = None
    else:
        trace_filter = st.selectbox("Filter by trace mode", ["all", "simple", "deep"])
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
            st.caption(f"Log: `{run.log_path.name}`")
            st.caption(f"Env: `{run.env}`")
            st.caption(f"Trace: `{run.trace_mode}`")
            st.caption(f"Epoch CSV: `{'✓' if has_csv else '—'}`")
            st.caption(f"Batch CSV: `{'✓' if run.batch_csv_path else '—'}`")
            st.caption(f"Per-class CSV: `{'✓' if run.perclass_csv_path else '—'}`")

    st.markdown("---")
    st.subheader("🟢 Live Monitor")
    live_mode = st.toggle("Auto-refresh activo", key="live_mode")
    refresh_interval = st.slider("Intervalo (s)", 5, 60, 10, disabled=not live_mode)

# ── Tabs ──────────────────────────────────────────────────────────────────────

(
    tab_curves, tab_perclass, tab_batch, tab_compare,
    tab_feasibility, tab_time, tab_info,
    tab_launcher, tab_live,
) = st.tabs([
    "📉 Training Curves", "🎯 Per-class Metrics", "📊 Batch Monitor",
    "🔀 Compare Runs", "⚡ Feasibility", "⏱ Time Analysis", "ℹ️ Run Info",
    "🚀 Launcher", "🟢 Live Monitor",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1: Training Curves
# ═══════════════════════════════════════════════════════════════════════════════

with tab_curves:
    if selected_run is None:
        st.info("Selecciona un run en el sidebar.")
    else:
        df = _load_df(
            str(run.log_path),
            str(run.epoch_csv_path) if run.epoch_csv_path else None,
        )

        if df.empty:
            st.error("No se pudieron parsear epochs del run seleccionado.")
        else:
            n_epochs = len(df)
            best_f1 = df["val_f1"].max() if "val_f1" in df.columns else float("nan")
            best_epoch = int(df.loc[df["val_f1"].idxmax(), "epoch"]) if not pd.isna(best_f1) else "—"
            best_thresh_f1 = (
                df["f1_at_threshold"].max()
                if "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any()
                else None
            )

            # Metric cards
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Epochs", n_epochs)
            c2.metric("Mejor Val F1", f"{best_f1:.4f}" if not pd.isna(best_f1) else "—")
            c3.metric("Mejor epoch", best_epoch)
            if best_thresh_f1 is not None:
                c4.metric("F1 @ threshold óptimo", f"{best_thresh_f1:.4f}")
            if "epoch_time" in df.columns and df["epoch_time"].notna().any():
                total_s = df["epoch_time"].sum()
                c5.metric("Duración total", f"{int(total_s//3600)}h {int((total_s%3600)//60)}m")

            if run.epoch_csv_path and run.epoch_csv_path.exists():
                st.caption("📄 Fuente: epoch_metrics CSV")
            else:
                st.caption("📄 Fuente: log file (sin CSV — usa el trainer actual para generarlo)")

            # F1 — con línea de threshold óptimo si está disponible
            extra_thresh: list = []
            if "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any():
                extra_thresh = [go.Scatter(
                    x=df["epoch"], y=df["f1_at_threshold"],
                    name="F1 @ threshold óptimo", mode="lines",
                    line=dict(color=COLORS[2], width=2, dash="dot"),
                )]

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    _metric_fig(df, "train_f1", "val_f1", "F1 Macro", "F1",
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

            # Epoch time bar chart
            if "epoch_time" in df.columns and df["epoch_time"].notna().any():
                et = df[["epoch", "epoch_time"]].dropna()
                fig_et = go.Figure(go.Bar(
                    x=et["epoch"], y=et["epoch_time"] / 60,
                    name="Tiempo/epoch (min)",
                    marker_color=COLORS[0],
                ))
                fig_et.update_layout(
                    title="Tiempo por epoch", xaxis_title="Epoch",
                    yaxis_title="Minutos", height=280,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_et, use_container_width=True)

            # Downloadable table
            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Descargar epoch_metrics.csv", csv_bytes,
                file_name="epoch_metrics.csv", mime="text/csv",
            )

            with st.expander("Tabla de métricas completa"):
                st.dataframe(df.set_index("epoch"), use_container_width=True)

            if run.plot_path and run.plot_path.exists():
                with st.expander("PNG guardado"):
                    st.image(str(run.plot_path), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2: Per-class Metrics
# ═══════════════════════════════════════════════════════════════════════════════

with tab_perclass:
    if selected_run is None:
        st.info("Selecciona un run en el sidebar.")
    else:
        subtab_bars, subtab_trend, subtab_cm = st.tabs(
            ["Métricas por clase", "Tendencia por clase", "Matriz de confusión"]
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

                # Color-coded ranking table
                def _color_f1(v: float) -> str:
                    if v >= 0.6:
                        return "background-color: #d4edda"
                    if v >= 0.3:
                        return "background-color: #fff3cd"
                    return "background-color: #f8d7da"

                styled = ep_df[["class_name", "f1", "precision", "recall"]].style.applymap(
                    _color_f1, subset=["f1"]
                ).format({"f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"})
                st.dataframe(styled, use_container_width=True, height=300)

                # Grouped bar chart
                colors_f1 = [
                    COLORS[2] if v >= 0.6 else (COLORS[1] if v >= 0.3 else COLORS[3])
                    for v in ep_df["f1"]
                ]
                fig_pc = go.Figure()
                fig_pc.add_trace(go.Bar(
                    y=ep_df["class_name"], x=ep_df["precision"],
                    name="Precision", orientation="h", marker_color=COLORS[0], opacity=0.85,
                ))
                fig_pc.add_trace(go.Bar(
                    y=ep_df["class_name"], x=ep_df["recall"],
                    name="Recall", orientation="h", marker_color=COLORS[1], opacity=0.85,
                ))
                fig_pc.add_trace(go.Bar(
                    y=ep_df["class_name"], x=ep_df["f1"],
                    name="F1", orientation="h", marker_color=colors_f1,
                ))
                fig_pc.update_layout(
                    barmode="group", title=f"Per-class Metrics — Epoch {selected_ep}",
                    xaxis_title="Score", yaxis_title="Clase",
                    height=620, margin=dict(l=200, r=20, t=50, b=40),
                    xaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(fig_pc, use_container_width=True)

            elif run.perclass_paths:
                st.caption("Mostrando PNGs estáticos (no hay perclass CSV)")
                epoch_opts = [p.stem.split("_epoch")[-1] for p in run.perclass_paths]
                idx = st.selectbox("Epoch", range(len(run.perclass_paths)),
                                   format_func=lambda i: f"Epoch {epoch_opts[i]}")
                if run.perclass_paths[idx].exists():
                    st.image(Image.open(run.perclass_paths[idx]), use_container_width=True)
            else:
                st.info("Sin datos por clase. Usa `--layers confusion` para generarlos.")

        with subtab_trend:
            if run.perclass_csv_path and run.perclass_csv_path.exists():
                pcdf = _load_perclass(str(run.perclass_csv_path))
                classes = sorted(pcdf["class_name"].unique().tolist())

                selected_classes = st.multiselect(
                    "Clases a comparar (máx. 8)", classes,
                    default=classes[:4],
                    max_selections=8,
                )
                metric_sel = st.radio("Métrica", ["f1", "precision", "recall"], horizontal=True)

                if selected_classes:
                    fig_trend = go.Figure()
                    for i, cls in enumerate(selected_classes):
                        cdf = pcdf[pcdf["class_name"] == cls].sort_values("epoch")
                        fig_trend.add_trace(go.Scatter(
                            x=cdf["epoch"], y=cdf[metric_sel],
                            name=cls[:30], mode="lines+markers",
                            line=dict(color=COLORS[i % len(COLORS)], width=2),
                            marker=dict(size=5),
                        ))
                    fig_trend.update_layout(
                        title=f"{metric_sel.upper()} por clase a lo largo de los epochs",
                        xaxis_title="Epoch", yaxis_title=metric_sel.capitalize(),
                        height=420, yaxis=dict(range=[0, 1]),
                        margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Sin perclass CSV disponible para este run.")

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
                    cm_mode = st.radio("Modo", ["Normalizada", "Absoluta"], key="cm_mode")

                pivot = get_matrix_for_epoch(cm_df, selected_cm_ep)
                class_order = list(pivot.index)
                z_norm = pivot.reindex(index=class_order, columns=class_order).values

                if cm_mode == "Absoluta":
                    row_sums = z_norm.sum(axis=1, keepdims=True)
                    z_abs = (z_norm * row_sums).round().astype(int)
                    z_plot = z_abs.tolist()
                    text = [[str(v) if v > 0 else "" for v in row] for row in z_abs]
                    zmin, zmax = 0, None
                    colorbar_title = "Muestras"
                else:
                    z_plot = z_norm.tolist()
                    text = [[f"{v:.2f}" if v >= 0.05 else "" for v in row] for row in z_norm]
                    zmin, zmax = 0, 1
                    colorbar_title = "P(pred j | true i)"

                fig_cm = go.Figure(go.Heatmap(
                    z=z_plot, x=class_order, y=class_order,
                    colorscale="Blues", zmin=zmin, zmax=zmax,
                    text=text, texttemplate="%{text}",
                    textfont={"size": 8},
                    hovertemplate="Verdadero: %{y}<br>Predicho: %{x}<br>valor: %{z:.3f}<extra></extra>",
                    colorbar=dict(title=colorbar_title),
                ))
                fig_cm.update_layout(
                    title=f"Matriz de confusión ({cm_mode}) — Epoch {selected_cm_ep}",
                    xaxis=dict(title="Clase predicha", tickangle=45, tickfont=dict(size=9)),
                    yaxis=dict(title="Clase verdadera", tickfont=dict(size=9), autorange="reversed"),
                    height=650, margin=dict(l=160, r=20, t=60, b=160),
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption("Diagonal = recall por clase. Fuera de diagonal = confusiones entre clases.")

            elif run.confusion_matrix_paths:
                st.caption("Mostrando PNG estático")
                epoch_labels = [p.stem.split("_epoch")[-1] for p in run.confusion_matrix_paths]
                cm_idx = st.selectbox(
                    "Epoch", range(len(run.confusion_matrix_paths)),
                    format_func=lambda i: f"Epoch {epoch_labels[i]}",
                    key="cm_epoch_sel",
                )
                if run.confusion_matrix_paths[cm_idx].exists():
                    st.image(Image.open(run.confusion_matrix_paths[cm_idx]), use_container_width=True)
            else:
                st.info("Sin matriz de confusión. Usa `--layers confusion` para generarla.")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3: Batch Monitor
# ═══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    if selected_run is None:
        st.info("Selecciona un run en el sidebar.")
    elif not run.batch_csv_path:
        st.info("Sin CSV de batch para este run. Usa `--layers batch-monitor` para generarlo.")
    else:
        bdf = _load_batch(str(run.batch_csv_path))
        epochs_available = sorted(bdf["epoch"].unique())

        col_ep, col_ma, col_info = st.columns([2, 2, 1])
        with col_ep:
            selected_epochs = st.multiselect(
                "Epochs", epochs_available, default=list(epochs_available[:3]),
            )
        with col_ma:
            ma_window = st.slider("Ventana moving average (batches)", 0, 200, 20,
                                  help="0 = desactivado")
        with col_info:
            if not bdf.empty:
                n_batches = int(bdf["n_batches"].iloc[0])
                batch_size_info = "—"
                st.caption(f"Batches/epoch: {n_batches}")

        if selected_epochs:
            fig = go.Figure()
            for i, ep in enumerate(selected_epochs):
                subset = bdf[bdf["epoch"] == ep].copy()
                color = COLORS[i % len(COLORS)]

                # Raw line
                fig.add_trace(go.Scatter(
                    x=subset["batch"], y=subset["running_loss"],
                    name=f"Epoch {ep}", mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.5, legendgroup=f"ep{ep}",
                ))

                if ma_window > 0 and len(subset) >= ma_window:
                    ma = subset["running_loss"].rolling(ma_window, center=True).mean()
                    fig.add_trace(go.Scatter(
                        x=subset["batch"], y=ma,
                        name=f"Epoch {ep} (MA{ma_window})", mode="lines",
                        line=dict(color=color, width=2.5),
                        legendgroup=f"ep{ep}",
                    ))

                    # Spike detection: batch where loss > mean + 2*std
                    mean_l = subset["running_loss"].mean()
                    std_l = subset["running_loss"].std()
                    spikes = subset[subset["running_loss"] > mean_l + 2 * std_l]
                    if not spikes.empty:
                        fig.add_trace(go.Scatter(
                            x=spikes["batch"], y=spikes["running_loss"],
                            name=f"Pico Ep{ep}", mode="markers",
                            marker=dict(color="red", size=8, symbol="x"),
                            legendgroup=f"ep{ep}", showlegend=False,
                        ))

            fig.update_layout(
                title="Running Loss por Batch",
                xaxis_title="Batch", yaxis_title="Running Loss",
                height=430, margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw batch data"):
                st.dataframe(bdf[bdf["epoch"].isin(selected_epochs)], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4: Compare Runs
# ═══════════════════════════════════════════════════════════════════════════════

with tab_compare:
    if not runs:
        st.info("Sin runs disponibles.")
    else:
        all_run_labels = {r.label: r for r in runs}
        all_labels_list = list(all_run_labels.keys())

        selected_compare = st.multiselect(
            "Runs a comparar (máx. 4)", all_labels_list,
            default=all_labels_list[:min(2, len(all_labels_list))],
            max_selections=4,
        )

        if len(selected_compare) < 2:
            st.info("Selecciona al menos 2 runs.")
        else:
            compare_runs = [(lbl, all_run_labels[lbl]) for lbl in selected_compare]
            compare_dfs: list[tuple[str, pd.DataFrame]] = []
            for lbl, r in compare_runs:
                cdf = _load_df(
                    str(r.log_path),
                    str(r.epoch_csv_path) if r.epoch_csv_path else None,
                )
                compare_dfs.append((lbl[:30], cdf))

            # Summary comparison table
            summary_rows = []
            for lbl, r in compare_runs:
                cdf = next(d for l, d in compare_dfs if l == lbl[:30])
                best_f1 = cdf["val_f1"].max() if "val_f1" in cdf.columns else float("nan")
                best_ep = int(cdf.loc[cdf["val_f1"].idxmax(), "epoch"]) if not pd.isna(best_f1) else "—"
                final_f1 = cdf["val_f1"].iloc[-1] if "val_f1" in cdf.columns else float("nan")
                total_s = cdf["epoch_time"].sum() if "epoch_time" in cdf.columns else float("nan")
                dur = f"{int(total_s//3600)}h {int((total_s%3600)//60)}m" if not pd.isna(total_s) else "—"
                summary_rows.append({
                    "Run": lbl[:40],
                    "Mejor Val F1": f"{best_f1:.4f}" if not pd.isna(best_f1) else "—",
                    "Mejor Epoch": best_ep,
                    "F1 Final": f"{final_f1:.4f}" if not pd.isna(final_f1) else "—",
                    "Epochs": len(cdf),
                    "Duración": dur,
                    "Env": r.env,
                    "Trace": r.trace_mode,
                })

            st.subheader("Tabla comparativa")
            st.dataframe(pd.DataFrame(summary_rows).set_index("Run"), use_container_width=True)

            # Metric charts
            metrics_to_compare = st.multiselect(
                "Métricas a superponer",
                ["val_f1", "val_loss", "train_f1", "train_loss", "val_prec", "val_rec", "epoch_time"],
                default=["val_f1", "val_loss"],
            )

            n_cols = 2
            cols = st.columns(n_cols)
            for idx, col_name in enumerate(metrics_to_compare):
                y_label = col_name.replace("_", " ")
                fig = _overlay_fig(compare_dfs, col=col_name, title=col_name, y_label=y_label)
                cols[idx % n_cols].plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5: Feasibility
# ═══════════════════════════════════════════════════════════════════════════════

with tab_feasibility:
    subtab_report, subtab_launcher_feas = st.tabs(["📋 Informe", "▶ Ejecutar Feasibility"])

    with subtab_report:
        feasibility_csvs = _get_feasibility_csvs()

        if not feasibility_csvs:
            st.info("No hay CSVs de feasibility. Ejecuta el feasibility check para generarlos.")
        else:
            csv_labels = {str(p): f"{p.parent.name}/{p.name}" for p in feasibility_csvs}
            selected_feas_path = st.selectbox(
                "Informe de feasibility", list(csv_labels.keys()),
                format_func=lambda p: csv_labels[p],
            )
            meta, bdf_feas = parse_feasibility_csv(Path(selected_feas_path))

            # ── Sección Modelo ──
            if meta:
                st.subheader("Modelo")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Modelo", meta.get("model_name", "—"))
                mc2.metric("Parámetros (M)", meta.get("total_params_M", "—"))
                mc3.metric("FLOPs (MFLOPs)", meta.get("flops_mflops", "—"))
                mc4.metric("Hardware", meta.get("hardware_name", "—"))

                # Memoria del modelo
                mem_keys = ["weight_mb", "gradient_mb", "optimizer_mb",
                            "activation_mb_per_image", "total_static_mb"]
                if any(k in meta for k in mem_keys):
                    st.subheader("Memoria estática del modelo")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Pesos (MB)", meta.get("weight_mb", "—"))
                    m2.metric("Gradientes (MB)", meta.get("gradient_mb", "—"))
                    m3.metric("AdamW (MB)", meta.get("optimizer_mb", "—"))
                    m4.metric("Activaciones/img (MB)", meta.get("activation_mb_per_image", "—"))
                    m5.metric("Total estático (MB)", meta.get("total_static_mb", "—"))

                # ── Sección Hardware ──
                st.subheader("Hardware")
                h1, h2, h3 = st.columns(3)
                total_vram = meta.get("total_vram_gb")
                free_vram = meta.get("free_vram_gb")
                h1.metric("VRAM total (GB)", total_vram if total_vram else "—")
                h2.metric("VRAM libre (GB)", free_vram if free_vram else "—")
                if total_vram and free_vram:
                    pct = float(free_vram) / float(total_vram) * 100
                    h3.metric("VRAM libre %", f"{pct:.1f}%")

                if total_vram and free_vram:
                    fig_vram_hw = go.Figure(go.Bar(
                        x=["Libre", "Ocupada"],
                        y=[float(free_vram), float(total_vram) - float(free_vram)],
                        marker_color=[COLORS[2], COLORS[3]],
                    ))
                    fig_vram_hw.update_layout(
                        title="Distribución VRAM", yaxis_title="GB",
                        height=220, margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_vram_hw, use_container_width=True)

            # ── Benchmark ──
            if not bdf_feas.empty:
                st.subheader("Benchmark — resultados completos")
                st.dataframe(bdf_feas, use_container_width=True)

                # Throughput chart train vs eval
                has_train_eval = ("imgs_per_s_train" in bdf_feas.columns
                                  and "imgs_per_s_eval" in bdf_feas.columns)
                has_legacy = "imgs_per_s" in bdf_feas.columns

                viable = bdf_feas[bdf_feas["oom"] == "no"].copy()
                if not viable.empty:
                    st.subheader("Throughput")
                    if has_train_eval:
                        fig_tp = go.Figure()
                        for mode in viable["trace_mode"].unique():
                            sub = viable[viable["trace_mode"] == mode]
                            x_labels = sub["batch_size"].astype(str) + f" [{mode}]"
                            fig_tp.add_trace(go.Bar(
                                x=x_labels, y=sub["imgs_per_s_train"],
                                name=f"Train [{mode}]",
                            ))
                            fig_tp.add_trace(go.Bar(
                                x=x_labels, y=sub["imgs_per_s_eval"],
                                name=f"Eval [{mode}]",
                            ))
                    else:
                        fig_tp = go.Figure()
                        for mode in viable["trace_mode"].unique():
                            sub = viable[viable["trace_mode"] == mode]
                            fig_tp.add_trace(go.Bar(
                                x=sub["batch_size"].astype(str), y=sub["imgs_per_s"],
                                name=f"trace={mode}",
                            ))

                    fig_tp.update_layout(
                        barmode="group", title="Throughput por batch size y trace mode",
                        xaxis_title="Batch size", yaxis_title="imgs/s",
                        height=350, margin=dict(l=40, r=20, t=50, b=40),
                    )
                    st.plotly_chart(fig_tp, use_container_width=True)

                    # VRAM chart
                    if "peak_vram_gb" in viable.columns and viable["peak_vram_gb"].notna().any():
                        st.subheader("VRAM por batch size")
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
                                annotation_text=f"VRAM disponible: {meta['free_vram_gb']} GB",
                                annotation_position="top left",
                            )
                        fig_vram.update_layout(
                            barmode="group", title="VRAM pico por batch size",
                            xaxis_title="Batch size", yaxis_title="GB",
                            height=300, margin=dict(l=40, r=20, t=50, b=40),
                        )
                        st.plotly_chart(fig_vram, use_container_width=True)

                # ── Estimaciones ──
                est_cols = [c for c in bdf_feas.columns if c.startswith("est_")]
                if est_cols and not viable.empty:
                    st.subheader("Estimaciones de tiempo")

                    # Selector de N epochs para recalcular
                    orig_ep_col = next((c for c in bdf_feas.columns
                                        if c.startswith("est_total_h_") and c.endswith("ep")), None)
                    orig_n = None
                    if orig_ep_col:
                        try:
                            orig_n = int(orig_ep_col.split("est_total_h_")[1].replace("ep", ""))
                        except ValueError:
                            pass

                    recalc_n = st.number_input(
                        "Epochs para recalcular estimación", min_value=1, value=orig_n or 30,
                        help="Recalcula est_total_h en base a los datos de throughput del CSV",
                    )

                    # Build display table
                    display_cols = ["batch_size", "trace_mode", "oom"]
                    for c in ["est_train_min_per_epoch", "est_eval_min_per_epoch",
                              "est_total_min_per_epoch"]:
                        if c in bdf_feas.columns:
                            display_cols.append(c)
                    if orig_ep_col:
                        display_cols.append(orig_ep_col)

                    est_df = bdf_feas[display_cols].copy()

                    # Recalc column
                    if "est_total_min_per_epoch" in bdf_feas.columns:
                        est_df[f"est_total_h_{recalc_n}ep (recalc)"] = (
                            bdf_feas["est_total_min_per_epoch"] * recalc_n / 60
                        ).round(2)

                    st.dataframe(est_df, use_container_width=True)

    with subtab_launcher_feas:
        st.subheader("Ejecutar Feasibility Check")
        configs_available = _get_configs()
        model_options = [
            "vit_base_patch16_224", "vit_tiny_patch16_224",
            "vit_small_patch16_224", "resnet50", "efficientnet_b0",
        ]
        with st.form("feasibility_form"):
            fa_col, fb_col = st.columns(2)
            with fa_col:
                feas_model = st.selectbox("Modelo", model_options)
                feas_batches = st.multiselect("Batch sizes", [16, 32, 64, 128], default=[32, 64])
                feas_epochs = st.number_input("Epochs para estimación", min_value=1, value=30)
            with fb_col:
                feas_traces = st.multiselect(
                    "Trace modes", ["off", "simple", "deep"], default=["off", "simple"],
                )
                feas_nfs = st.slider("NFS factor", 1.0, 2.0, 1.0, 0.05,
                                     help="Factor de corrección para latencia NFS (Verode: ~1.3)")
                if configs_available:
                    feas_config = st.selectbox("Config YAML (opcional)", ["(ninguno)"] + configs_available)
                else:
                    feas_config = "(ninguno)"
            submitted_feas = st.form_submit_button("▶ Ejecutar Feasibility")

        if submitted_feas:
            if not feas_batches:
                st.error("Selecciona al menos un batch size.")
            else:
                bs_args = " ".join(str(b) for b in feas_batches)
                trace_args = " ".join(feas_traces) if feas_traces else "off"
                cmd_parts = [
                    "uv run python scripts/check_feasibility.py",
                    f"--model {feas_model}",
                    f"--batch-sizes {bs_args}",
                    f"--epochs {feas_epochs}",
                    f"--trace-modes {trace_args}",
                ]
                if feas_nfs != 1.0:
                    cmd_parts.append(f"--nfs-factor {feas_nfs}")
                if feas_config != "(ninguno)":
                    cmd_parts.append(f"--config configs/{feas_config}")
                cmd = " ".join(cmd_parts)
                st.code(cmd, language="bash")
                out_placeholder = st.empty()
                with st.spinner("Ejecutando feasibility check…"):
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, cwd=str(ROOT),
                    )
                if result.returncode == 0:
                    st.success("Completado")
                    out_placeholder.code(
                        result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout
                    )
                    _get_feasibility_csvs.clear()
                else:
                    st.error("Error:")
                    out_placeholder.code(result.stderr[-2000:])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 6: Time Analysis
# ═══════════════════════════════════════════════════════════════════════════════

with tab_time:
    if selected_run is None:
        st.info("Selecciona un run en el sidebar.")
    else:
        df_time = _load_df(
            str(run.log_path),
            str(run.epoch_csv_path) if run.epoch_csv_path else None,
        )

        if "epoch_time" not in df_time.columns or df_time["epoch_time"].isna().all():
            st.info("Sin datos de tiempo por epoch. Usa `--trace simple` para generarlos.")
        else:
            et = df_time[["epoch", "epoch_time"]].dropna()

            # Summary metrics
            t1, t2, t3, t4 = st.columns(4)
            total_s = et["epoch_time"].sum()
            avg_s = et["epoch_time"].mean()
            t1.metric("Total", f"{int(total_s//3600)}h {int((total_s%3600)//60)}m")
            t2.metric("Avg/epoch", f"{avg_s/60:.1f} min")
            t3.metric("Min/epoch", f"{et['epoch_time'].min()/60:.1f} min")
            t4.metric("Max/epoch", f"{et['epoch_time'].max()/60:.1f} min")

            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=et["epoch"], y=et["epoch_time"] / 60,
                name="Tiempo real (min)", mode="lines+markers",
                line=dict(color=COLORS[0], width=2), marker=dict(size=5),
            ))

            # Train vs eval breakdown if available
            has_train_time = ("epoch_time_train_s" in df_time.columns
                              and df_time["epoch_time_train_s"].notna().any())
            has_eval_time = ("epoch_time_eval_s" in df_time.columns
                             and df_time["epoch_time_eval_s"].notna().any())
            if has_train_time:
                fig_time.add_trace(go.Scatter(
                    x=et["epoch"], y=df_time["epoch_time_train_s"] / 60,
                    name="Train (min)", mode="lines",
                    line=dict(color=COLORS[2], width=2, dash="dot"),
                ))
            if has_eval_time:
                fig_time.add_trace(go.Scatter(
                    x=et["epoch"], y=df_time["epoch_time_eval_s"] / 60,
                    name="Eval (min)", mode="lines",
                    line=dict(color=COLORS[1], width=2, dash="dash"),
                ))

            # Linear trend
            x_arr = et["epoch"].values.astype(float)
            y_arr = et["epoch_time"].values / 60
            if len(x_arr) >= 2:
                coeffs = np.polyfit(x_arr, y_arr, 1)
                y_trend = np.polyval(coeffs, x_arr)
                fig_time.add_trace(go.Scatter(
                    x=et["epoch"], y=y_trend,
                    name="Tendencia lineal", mode="lines",
                    line=dict(color="gray", width=1, dash="dash"),
                ))

            # Warmup detection: check config for warmup_epochs
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
                    fillcolor="orange", opacity=0.08,
                    annotation_text=f"Warmup ({warmup_ep}ep)",
                    annotation_position="top left",
                )

            # Overlay feasibility estimate
            feasibility_csvs = _get_feasibility_csvs()
            if feasibility_csvs:
                try:
                    _, bdf_t = parse_feasibility_csv(feasibility_csvs[0])
                    est_col = next(
                        (c for c in bdf_t.columns if "est_total_min_per_epoch" in c), None
                    )
                    if est_col and not bdf_t.empty:
                        viable_t = bdf_t[(bdf_t["oom"] == "no") & bdf_t[est_col].notna()]
                        if not viable_t.empty:
                            tp_col = "imgs_per_s_train" if "imgs_per_s_train" in viable_t else "imgs_per_s"
                            if tp_col in viable_t:
                                best_row = viable_t.loc[viable_t[tp_col].idxmax()]
                                est_min = float(best_row[est_col])
                                fig_time.add_hline(
                                    y=est_min, line_dash="dash", line_color=COLORS[1],
                                    annotation_text=f"Estimación feasibility: {est_min:.0f} min/epoch",
                                    annotation_position="top right",
                                )
                except Exception:
                    pass

            fig_time.update_layout(
                title="Tiempo por epoch",
                xaxis_title="Epoch", yaxis_title="Minutos",
                height=400, margin=dict(l=40, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # Estimated vs real comparison table
            if feasibility_csvs:
                try:
                    _, bdf_comp = parse_feasibility_csv(feasibility_csvs[0])
                    est_col2 = next(
                        (c for c in bdf_comp.columns if "est_total_min_per_epoch" in c), None
                    )
                    if est_col2:
                        viable_comp = bdf_comp[(bdf_comp["oom"] == "no") & bdf_comp[est_col2].notna()]
                        if not viable_comp.empty:
                            tp_col2 = "imgs_per_s_train" if "imgs_per_s_train" in viable_comp else "imgs_per_s"
                            if tp_col2 in viable_comp:
                                best_comp = viable_comp.loc[viable_comp[tp_col2].idxmax()]
                                est_val = float(best_comp[est_col2])
                                real_val = avg_s / 60
                                err_pct = (real_val - est_val) / est_val * 100 if est_val else 0
                                st.subheader("Estimado vs Real")
                                ce1, ce2, ce3 = st.columns(3)
                                ce1.metric("Estimado (min/epoch)", f"{est_val:.1f}")
                                ce2.metric("Real avg (min/epoch)", f"{real_val:.1f}")
                                ce3.metric("Error relativo", f"{err_pct:+.1f}%")
                except Exception:
                    pass

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 7: Run Info
# ═══════════════════════════════════════════════════════════════════════════════

with tab_info:
    if selected_run is None:
        st.info("Selecciona un run en el sidebar.")
    else:
        df_info = _load_df(
            str(run.log_path),
            str(run.epoch_csv_path) if run.epoch_csv_path else None,
        )
        n_ep_info = len(df_info)
        best_f1_info = df_info["val_f1"].max() if "val_f1" in df_info.columns else float("nan")
        best_ep_info = (
            int(df_info.loc[df_info["val_f1"].idxmax(), "epoch"])
            if not pd.isna(best_f1_info) else "—"
        )

        col_meta, col_files = st.columns(2)

        with col_meta:
            st.subheader("Metadatos del run")
            rows = {
                "Log": run.log_path.name,
                "Env": run.env,
                "Trace mode": run.trace_mode,
                "Epochs": n_ep_info,
                "Mejor Val F1": f"{best_f1_info:.4f}" if not pd.isna(best_f1_info) else "—",
                "Mejor epoch": best_ep_info,
            }
            if "epoch_time" in df_info.columns and df_info["epoch_time"].notna().any():
                total_s = df_info["epoch_time"].sum()
                rows["Duración total"] = f"{int(total_s//3600)}h {int((total_s%3600)//60)}m"
                rows["Avg/epoch"] = f"{df_info['epoch_time'].mean()/60:.1f} min"
            for k, v in rows.items():
                st.markdown(f"**{k}:** {v}")

        with col_files:
            st.subheader("Archivos asociados")
            for label, path in [
                ("Plot", run.plot_path),
                ("Batch CSV", run.batch_csv_path),
                ("Per-class CSV", run.perclass_csv_path),
                ("Epoch CSV", run.epoch_csv_path),
            ]:
                val = path.name if path else "—"
                st.markdown(f"- **{label}:** `{val}`")
            for p in run.perclass_paths:
                st.markdown(f"- Per-class PNG: `{p.name}`")
            for p in run.confusion_matrix_paths:
                st.markdown(f"- CM PNG: `{p.name}`")

        # Config YAML detection
        st.subheader("Config YAML")
        configs_found: list[Path] = []
        for cfg in _get_configs():
            cfg_path = ROOT / "configs" / cfg
            try:
                import yaml
                cfg_data = yaml.safe_load(cfg_path.read_text())
                env_cfg = cfg_data.get("output", {}).get("env", "")
                if env_cfg == run.env or (run.env == "local" and "cluster" not in cfg):
                    configs_found.append(cfg_path)
            except Exception:
                pass
        if configs_found:
            cfg_sel = st.selectbox("Config detectada", [p.name for p in configs_found])
            cfg_path_sel = next(p for p in configs_found if p.name == cfg_sel)
            st.code(cfg_path_sel.read_text(), language="yaml")
        else:
            st.info("No se pudo determinar el config de este run.")

        # Anomaly summary
        anomalies = _detect_anomalies_in_log(run.log_path)
        if anomalies:
            st.subheader(f"⚠️ Anomalías detectadas en el log ({len(anomalies)})")
            with st.expander("Ver anomalías"):
                for line in anomalies:
                    st.text(line)
        else:
            st.success("✅ Sin anomalías detectadas en el log.")

        # Full searchable log
        st.subheader("Log completo")
        search_term = st.text_input("🔍 Buscar en el log", "")
        try:
            all_lines = run.log_path.read_text(errors="replace").splitlines()
            if search_term:
                filtered_lines = [l for l in all_lines if search_term.lower() in l.lower()]
                st.caption(f"{len(filtered_lines)} / {len(all_lines)} líneas")
                st.code("\n".join(filtered_lines[-300:]), language="text")
            else:
                st.caption(f"{len(all_lines)} líneas totales")
                st.code("\n".join(all_lines[-300:]), language="text")
        except Exception as exc:
            st.error(str(exc))

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 8: Launcher
# ═══════════════════════════════════════════════════════════════════════════════

with tab_launcher:
    subtab_single, subtab_ddp = st.tabs(["🖥 Single GPU", "🔗 DDP (multi-GPU)"])

    configs_available = _get_configs()
    model_options_launcher = [
        "vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224",
        "resnet50", "efficientnet_b0", "deit_tiny_patch16_224",
    ]

    with subtab_single:
        st.subheader("Lanzar entrenamiento Single GPU")

        with st.form("launcher_single_form"):
            la1, la2 = st.columns(2)
            with la1:
                l_model = st.selectbox("Modelo", model_options_launcher)
                l_config = st.selectbox(
                    "Config YAML", configs_available if configs_available else ["(ninguno)"]
                )
                l_epochs = st.number_input("Epochs override", min_value=0, value=0,
                                           help="0 = usar el valor del config")
                l_batch_size = st.number_input("Batch size override", min_value=0, value=0,
                                               help="0 = usar el valor del config")
            with la2:
                l_trace = st.selectbox("Trace mode", ["simple", "off", "deep"])
                l_layers = st.multiselect(
                    "Layers", ["plot", "hooks", "confusion", "batch-monitor"],
                    default=["plot", "confusion"],
                )
                l_fn = st.multiselect("Fn decorators", ["timing", "energy"])
                l_inspect = st.multiselect(
                    "Inspect (deep features)",
                    ["model-summary", "grad-monitor", "anomalies", "batch-table"],
                )

            launched_single = st.form_submit_button("🚀 Lanzar entrenamiento")

        if launched_single:
            cmd_parts = ["uv run python scripts/train_single_gpu.py"]
            cmd_parts.append(f"--config configs/{l_config}")
            if l_model:
                cmd_parts.append(f"--model {l_model}")
            if l_epochs > 0:
                cmd_parts.append(f"--epochs {l_epochs}")
            if l_batch_size > 0:
                cmd_parts.append(f"--batch-size {l_batch_size}")
            cmd_parts.append(f"--trace {l_trace}")
            if l_layers:
                cmd_parts.append(f"--layers {' '.join(l_layers)}")
            if l_fn:
                cmd_parts.append(f"--fn {' '.join(l_fn)}")
            if l_inspect:
                cmd_parts.append(f"--inspect {' '.join(l_inspect)}")

            cmd = " ".join(cmd_parts)
            st.code(cmd, language="bash")

            out_placeholder = st.empty()
            rc = _launch_process(cmd, out_placeholder)

            if rc == 0:
                st.success("Entrenamiento completado.")
                _get_runs.clear()
            else:
                st.error(f"El proceso terminó con código {rc}.")

    with subtab_ddp:
        st.subheader("Lanzar entrenamiento DDP (multi-GPU)")

        with st.form("launcher_ddp_form"):
            dd1, dd2 = st.columns(2)
            with dd1:
                d_nproc = st.number_input("GPUs (--nproc_per_node)", min_value=1, max_value=8, value=2)
                d_model = st.selectbox("Modelo", model_options_launcher, key="ddp_model")
                d_config = st.selectbox(
                    "Config YAML", configs_available if configs_available else ["(ninguno)"],
                    key="ddp_config",
                )
                d_epochs = st.number_input("Epochs override", min_value=0, value=0, key="ddp_ep")
            with dd2:
                d_trace = st.selectbox("Trace mode", ["simple", "off", "deep"], key="ddp_trace")
                d_layers = st.multiselect(
                    "Layers", ["plot", "confusion", "batch-monitor"],
                    default=["plot"], key="ddp_layers",
                )
                d_fn = st.multiselect("Fn decorators", ["timing", "energy"], key="ddp_fn")

            launched_ddp = st.form_submit_button("🚀 Lanzar DDP")

        if launched_ddp:
            cmd_parts = [
                f"torchrun --nproc_per_node={d_nproc} scripts/train_ddp.py",
                f"--config configs/{d_config}",
                f"--model {d_model}",
                f"--trace {d_trace}",
            ]
            if d_epochs > 0:
                cmd_parts.append(f"--epochs {d_epochs}")
            if d_layers:
                cmd_parts.append(f"--layers {' '.join(d_layers)}")
            if d_fn:
                cmd_parts.append(f"--fn {' '.join(d_fn)}")

            cmd = " ".join(cmd_parts)
            st.code(cmd, language="bash")

            out_placeholder = st.empty()
            rc = _launch_process(cmd, out_placeholder)

            if rc == 0:
                st.success("Entrenamiento DDP completado.")
                _get_runs.clear()
            else:
                st.error(f"El proceso terminó con código {rc}.")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 9: Live Monitor
# ═══════════════════════════════════════════════════════════════════════════════

with tab_live:
    st.subheader("Monitor en tiempo real")

    # Filter runs modified in last 30 min
    import time as _time
    now_ts = _time.time()
    recent_runs = [
        r for r in runs
        if r.log_path.exists() and (now_ts - r.log_path.stat().st_mtime) < 1800
    ]

    if not recent_runs:
        st.info(
            "No hay runs activos (ningún log modificado en los últimos 30 min). "
            "Lanza un entrenamiento desde la pestaña Launcher."
        )
    else:
        live_labels = {r.label: r for r in recent_runs}
        live_sel_label = st.selectbox("Run activo", list(live_labels.keys()), key="live_run_sel")
        live_run = live_labels[live_sel_label]

        # GPU stats
        gpu = _gpu_usage()
        if gpu:
            gv1, gv2, gv3, gv4 = st.columns(4)
            gv1.metric("GPU", gpu["name"])
            gv2.metric("VRAM usada", f"{gpu['mem_used_mb']/1024:.1f} GB / {gpu['mem_total_mb']/1024:.1f} GB")
            gv3.metric("Utilización", f"{gpu['util_pct']}%")
            gv4.metric("Temperatura", f"{gpu['temp_c']} °C")
        else:
            st.caption("GPU info no disponible (nvidia-smi no encontrado).")

        # Epoch progress
        progress = _parse_log_progress(live_run.log_path)
        if progress["epochs"] > 0:
            pct = progress["epoch"] / progress["epochs"]
            st.progress(pct, text=f"Epoch {progress['epoch']} / {progress['epochs']}")
        if progress["last_val_f1"] is not None:
            m1, m2 = st.columns(2)
            m1.metric("Último Val F1", f"{progress['last_val_f1']:.4f}")
            if progress["last_val_loss"] is not None:
                m2.metric("Último Val Loss", f"{progress['last_val_loss']:.4f}")

        # Live epoch chart (from epoch CSV if available)
        if live_run.epoch_csv_path and live_run.epoch_csv_path.exists():
            live_df = _load_df(str(live_run.log_path), str(live_run.epoch_csv_path))
            if not live_df.empty:
                fig_live = go.Figure()
                if "val_f1" in live_df.columns:
                    fig_live.add_trace(go.Scatter(
                        x=live_df["epoch"], y=live_df["val_f1"],
                        name="Val F1", mode="lines+markers",
                        line=dict(color=COLORS[0], width=2),
                    ))
                if "val_loss" in live_df.columns:
                    fig_live.add_trace(go.Scatter(
                        x=live_df["epoch"], y=live_df["val_loss"],
                        name="Val Loss", mode="lines+markers",
                        line=dict(color=COLORS[1], width=2),
                    ))
                fig_live.update_layout(
                    title="Métricas en vivo", xaxis_title="Epoch",
                    height=300, margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_live, use_container_width=True)

        # Last N lines of log
        st.subheader("Últimas líneas del log")
        log_tail = _read_log_tail(live_run.log_path, n=40)
        st.code(log_tail, language="text")

    # Auto-refresh
    if live_mode:
        time.sleep(refresh_interval)
        _load_df.clear()
        st.rerun()
