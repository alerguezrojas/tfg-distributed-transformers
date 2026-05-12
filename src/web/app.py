"""Streamlit web dashboard for visualizing training runs."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.web.batch_parser import parse_batch_csv
from src.web.log_parser import parse_log
from src.web.run_registry import RunInfo, discover_runs

ROOT = Path(__file__).resolve().parents[2]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Training Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_df(log_path: str) -> pd.DataFrame:
    return parse_log(Path(log_path))


@st.cache_data(ttl=30)
def _load_batch(csv_path: str) -> pd.DataFrame:
    return parse_batch_csv(Path(csv_path))


@st.cache_data(ttl=60)
def _get_runs() -> list[RunInfo]:
    return discover_runs(ROOT)


def _metric_fig(
    df: pd.DataFrame,
    col_train: str,
    col_val: str,
    title: str,
    y_label: str,
    color_train: str = "#4c72b0",
    color_val: str = "#dd8452",
) -> go.Figure:
    fig = go.Figure()
    if col_train in df.columns and df[col_train].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_train],
            name="Train", mode="lines+markers",
            line=dict(color=color_train, width=2),
            marker=dict(size=5),
        ))
    if col_val in df.columns and df[col_val].notna().any():
        fig.add_trace(go.Scatter(
            x=df["epoch"], y=df[col_val],
            name="Val", mode="lines+markers",
            line=dict(color=color_val, width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title=title, xaxis_title="Epoch", yaxis_title=y_label,
        height=350, margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _overlay_fig(
    dfs: list[tuple[str, pd.DataFrame]],
    col: str,
    title: str,
    y_label: str,
) -> go.Figure:
    """Overlay a single column from multiple runs."""
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    fig = go.Figure()
    for i, (label, df) in enumerate(dfs):
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df["epoch"], y=df[col],
                name=label[:30], mode="lines+markers",
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=5),
            ))
    fig.update_layout(
        title=title, xaxis_title="Epoch", yaxis_title=y_label,
        height=380, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

runs = _get_runs()

with st.sidebar:
    st.title("Training Dashboard")
    st.markdown("---")

    if not runs:
        st.warning("No runs found in logs/.")
        st.stop()

    trace_filter = st.selectbox("Filter by trace mode", ["all", "simple", "deep"])
    filtered = [r for r in runs if trace_filter == "all" or r.trace_mode == trace_filter]

    if not filtered:
        st.warning("No runs match the filter.")
        st.stop()

    run_labels = {r.label: r for r in filtered}
    selected_label = st.selectbox("Run", list(run_labels.keys()))
    run = run_labels[selected_label]

    st.markdown("---")
    st.caption(f"Log: `{run.log_path.name}`")
    st.caption(f"Trace mode: `{run.trace_mode}`")
    st.caption(f"Plot: `{'yes' if run.plot_path else 'no'}`")
    st.caption(f"Batch CSV: `{'yes' if run.batch_csv_path else 'no'}`")
    st.caption(f"Per-class plots: `{len(run.perclass_paths)}`")

# ── Load data ─────────────────────────────────────────────────────────────────

df = _load_df(str(run.log_path))

if df.empty:
    st.error("Could not parse any epochs from the selected log file.")
    st.stop()

n_epochs = len(df)
best_f1 = df["val_f1"].max() if "val_f1" in df.columns else float("nan")
best_epoch = int(df.loc[df["val_f1"].idxmax(), "epoch"]) if not pd.isna(best_f1) else "—"

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Training Dashboard")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Epochs", n_epochs)
col2.metric("Best Val F1", f"{best_f1:.4f}" if not pd.isna(best_f1) else "—")
col3.metric("Best Epoch", best_epoch)
if "epoch_time" in df.columns and df["epoch_time"].notna().any():
    avg_time = df["epoch_time"].mean()
    col4.metric("Avg epoch time", f"{avg_time/60:.1f} min")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_curves, tab_perclass, tab_batch, tab_compare, tab_info = st.tabs([
    "Training Curves", "Per-class Metrics", "Batch Monitor", "Compare Runs", "Run Info",
])

# ── Tab: Training Curves ─────────────────────────────────────────────────────

with tab_curves:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            _metric_fig(df, "train_loss", "val_loss", "Loss", "BCE Loss"),
            use_container_width=True,
        )
        st.plotly_chart(
            _metric_fig(df, "val_prec", "val_rec", "Precision & Recall (val)", "Score",
                        color_train="#55a868", color_val="#c44e52"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            _metric_fig(df, "train_f1", "val_f1", "F1 Macro", "F1"),
            use_container_width=True,
        )
        st.plotly_chart(
            _metric_fig(df, "train_acc", "val_acc", "Accuracy", "Accuracy",
                        color_train="#8172b2", color_val="#937860"),
            use_container_width=True,
        )

    with st.expander("Raw epoch data"):
        st.dataframe(df.set_index("epoch"), use_container_width=True)

    if run.plot_path and run.plot_path.exists():
        st.subheader("Training plot (saved PNG)")
        st.image(str(run.plot_path), use_container_width=True)

# ── Tab: Per-class Metrics ───────────────────────────────────────────────────

with tab_perclass:
    if not run.perclass_paths:
        st.info("No per-class plots available for this run. Use `--layers confusion` to generate them.")
    else:
        epoch_options = [p.stem.split("_epoch")[-1] for p in run.perclass_paths]
        selected_epoch_idx = st.selectbox(
            "Epoch", range(len(run.perclass_paths)),
            format_func=lambda i: f"Epoch {epoch_options[i]}",
        )
        img_path = run.perclass_paths[selected_epoch_idx]
        if img_path.exists():
            img = Image.open(img_path)
            st.image(img, use_container_width=True)

# ── Tab: Batch Monitor ───────────────────────────────────────────────────────

with tab_batch:
    if not run.batch_csv_path:
        st.info("No batch-level CSV for this run. Use `--layers batch-monitor` to generate it.")
    else:
        bdf = _load_batch(str(run.batch_csv_path))
        epochs_available = sorted(bdf["epoch"].unique())
        selected_epochs = st.multiselect(
            "Epochs to display", epochs_available, default=epochs_available[:3],
        )

        fig = go.Figure()
        colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2",
                  "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]
        for i, ep in enumerate(selected_epochs):
            subset = bdf[bdf["epoch"] == ep]
            fig.add_trace(go.Scatter(
                x=subset["batch"], y=subset["running_loss"],
                name=f"Epoch {ep}", mode="lines",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig.update_layout(
            title="Running Loss per Batch",
            xaxis_title="Batch", yaxis_title="Running Loss",
            height=400, margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw batch data"):
            st.dataframe(bdf, use_container_width=True)

# ── Tab: Compare Runs ────────────────────────────────────────────────────────

with tab_compare:
    if len(filtered) < 2:
        st.info("Need at least 2 runs in the current filter to compare.")
    else:
        all_labels = list(run_labels.keys())
        default_a = all_labels[0]
        default_b = all_labels[1] if len(all_labels) > 1 else all_labels[0]

        col_a, col_b = st.columns(2)
        with col_a:
            label_a = st.selectbox("Run A", all_labels, index=0, key="cmp_a")
        with col_b:
            label_b = st.selectbox("Run B", all_labels, index=min(1, len(all_labels) - 1), key="cmp_b")

        run_a, run_b = run_labels[label_a], run_labels[label_b]
        df_a = _load_df(str(run_a.log_path))
        df_b = _load_df(str(run_b.log_path))

        short_a = label_a[:25]
        short_b = label_b[:25]
        pairs = [
            ("val_f1", "Val F1"),
            ("val_loss", "Val Loss"),
            ("train_f1", "Train F1"),
            ("train_loss", "Train Loss"),
        ]
        r1, r2 = st.columns(2)
        for i, (col, title) in enumerate(pairs):
            fig = _overlay_fig(
                [(short_a, df_a), (short_b, df_b)],
                col=col, title=title, y_label=col,
            )
            (r1 if i % 2 == 0 else r2).plotly_chart(fig, use_container_width=True)

# ── Tab: Run Info ────────────────────────────────────────────────────────────

with tab_info:
    st.subheader("Run metadata")

    info_rows = {
        "Log file": run.log_path.name,
        "Trace mode": run.trace_mode,
        "Epochs parsed": n_epochs,
        "Best Val F1": f"{best_f1:.4f}" if not pd.isna(best_f1) else "—",
        "Best epoch": best_epoch,
    }

    if "epoch_time" in df.columns and df["epoch_time"].notna().any():
        total_s = df["epoch_time"].sum()
        info_rows["Total training time"] = (
            f"{int(total_s // 3600)}h {int((total_s % 3600) // 60)}m"
        )
        info_rows["Avg epoch time"] = f"{df['epoch_time'].mean() / 60:.1f} min"

    for k, v in info_rows.items():
        st.markdown(f"**{k}:** {v}")

    st.subheader("Associated files")
    st.markdown(f"- Plot: `{run.plot_path.name if run.plot_path else '—'}`")
    st.markdown(f"- Batch CSV: `{run.batch_csv_path.name if run.batch_csv_path else '—'}`")
    if run.perclass_paths:
        for p in run.perclass_paths:
            st.markdown(f"- Per-class: `{p.name}`")
    else:
        st.markdown("- Per-class: —")

    with st.expander("View raw log (first 200 lines)"):
        lines = run.log_path.read_text(errors="replace").splitlines()[:200]
        st.code("\n".join(lines), language=None)
