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
            st.caption(
                f"**Log:** {run.log_path.name}  \n"
                f"**Env:** {run.env}  \n"
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
    tab_curves, tab_perclass, tab_batch, tab_compare,
    tab_feasibility, tab_time, tab_info,
    tab_launcher, tab_live,
) = st.tabs([
    "Curves", "Per-class", "Batch", "Compare",
    "Feasibility", "Time", "Info", "Launcher", "Live",
])

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
            best_f1 = df["val_f1"].max() if "val_f1" in df.columns else float("nan")
            best_epoch = int(df.loc[df["val_f1"].idxmax(), "epoch"]) if not pd.isna(best_f1) else "—"
            best_thresh_f1 = (
                df["f1_at_threshold"].max()
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
                    xaxis=dict(title="Predicted", tickangle=45, tickfont=dict(size=9)),
                    yaxis=dict(title="True", tickfont=dict(size=9), autorange="reversed"),
                    height=640, margin=dict(l=160, r=20, t=50, b=160),
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption("Diagonal = recall per class. Off-diagonal = confusion between classes.")

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
                best_f1_c = cdf["val_f1"].max() if ("val_f1" in cdf.columns and not cdf.empty) else float("nan")
                if not pd.isna(best_f1_c):
                    idx_c = cdf["val_f1"].idxmax()
                    best_ep_c = int(cdf.loc[idx_c, "epoch"]) if not pd.isna(idx_c) else "—"
                else:
                    best_ep_c = "—"
                _last = cdf["val_f1"].dropna() if ("val_f1" in cdf.columns and not cdf.empty) else pd.Series(dtype=float)
                final_f1_c = _last.iloc[-1] if not _last.empty else float("nan")
                total_s_c = cdf["epoch_time"].sum() if "epoch_time" in cdf.columns else float("nan")
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
    subtab_report, subtab_run_feas = st.tabs(["Report", "Run check"])

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
        best_f1_i = df_info["val_f1"].max() if "val_f1" in df_info.columns else float("nan")
        best_ep_i = (
            int(df_info.loc[df_info["val_f1"].idxmax(), "epoch"])
            if not pd.isna(best_f1_i) else "—"
        )

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
