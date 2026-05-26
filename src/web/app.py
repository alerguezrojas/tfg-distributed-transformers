"""Streamlit web dashboard for visualizing training runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.web.batch_parser import parse_batch_csv
from src.web.feasibility_parser import parse_feasibility_csv
from src.web.log_parser import parse_log
from src.web.perclass_parser import parse_perclass_csv
from src.web.run_registry import RunInfo, discover_feasibility_csvs, discover_runs

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
def _load_df(log_path: str, epoch_csv: str | None) -> pd.DataFrame:
    """Load epoch metrics — CSV-first, log fallback for older runs or empty CSVs."""
    if epoch_csv and Path(epoch_csv).exists():
        df = pd.read_csv(epoch_csv)
        if not df.empty:
            # Rename epoch_time_s to epoch_time for app compatibility
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
    has_csv = run.epoch_csv_path is not None and run.epoch_csv_path.exists()
    st.caption(f"Log: `{run.log_path.name}`")
    st.caption(f"Env: `{run.env}`")
    st.caption(f"Trace mode: `{run.trace_mode}`")
    st.caption(f"Epoch CSV: `{'✓' if has_csv else '—'}`")
    st.caption(f"Plot: `{'yes' if run.plot_path else 'no'}`")
    st.caption(f"Batch CSV: `{'yes' if run.batch_csv_path else 'no'}`")
    st.caption(f"Per-class CSV: `{'✓' if run.perclass_csv_path else '—'}`")

# ── Load data ─────────────────────────────────────────────────────────────────

df = _load_df(
    str(run.log_path),
    str(run.epoch_csv_path) if run.epoch_csv_path else None,
)

if df.empty:
    st.error("Could not parse any epochs from the selected run.")
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

tab_curves, tab_perclass, tab_batch, tab_compare, tab_feasibility, tab_time, tab_info = st.tabs([
    "Training Curves", "Per-class Metrics", "Batch Monitor",
    "Compare Runs", "Feasibility", "Time Analysis", "Run Info",
])

# ── Tab: Training Curves ─────────────────────────────────────────────────────

with tab_curves:
    if run.epoch_csv_path and run.epoch_csv_path.exists():
        st.caption("📄 Data source: epoch_metrics CSV")
    else:
        st.caption("📄 Data source: log file (no CSV found — run with current trainer to generate)")

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
    subtab_bars, subtab_cm = st.tabs(["Métricas por clase", "Matriz de confusión"])

    with subtab_bars:
        if run.perclass_csv_path and run.perclass_csv_path.exists():
            pcdf = _load_perclass(str(run.perclass_csv_path))
            epochs_available = sorted(pcdf["epoch"].unique().tolist())

            selected_ep = st.selectbox(
                "Epoch", epochs_available,
                format_func=lambda e: f"Epoch {e}",
            )
            ep_df = pcdf[pcdf["epoch"] == selected_ep].copy()
            ep_df = ep_df.sort_values("f1", ascending=False)

            colors_f1 = ["steelblue" if v >= 0.5 else "salmon" for v in ep_df["f1"]]

            fig_pc = go.Figure()
            fig_pc.add_trace(go.Bar(
                y=ep_df["class_name"], x=ep_df["precision"],
                name="Precision", orientation="h", marker_color="#4c72b0",
                opacity=0.85,
            ))
            fig_pc.add_trace(go.Bar(
                y=ep_df["class_name"], x=ep_df["recall"],
                name="Recall", orientation="h", marker_color="#dd8452",
                opacity=0.85,
            ))
            fig_pc.add_trace(go.Bar(
                y=ep_df["class_name"], x=ep_df["f1"],
                name="F1", orientation="h", marker_color=colors_f1,
            ))
            fig_pc.update_layout(
                barmode="group", title=f"Per-class Metrics — Epoch {selected_ep}",
                xaxis_title="Score", yaxis_title="Class",
                height=600, margin=dict(l=180, r=20, t=50, b=40),
                xaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig_pc, use_container_width=True)

            with st.expander("Raw per-class data"):
                st.dataframe(ep_df.set_index("class_name"), use_container_width=True)

        elif run.perclass_paths:
            st.caption("Showing static PNGs (no perclass CSV found)")
            epoch_options = [p.stem.split("_epoch")[-1] for p in run.perclass_paths]
            selected_epoch_idx = st.selectbox(
                "Epoch", range(len(run.perclass_paths)),
                format_func=lambda i: f"Epoch {epoch_options[i]}",
            )
            img_path = run.perclass_paths[selected_epoch_idx]
            if img_path.exists():
                st.image(Image.open(img_path), use_container_width=True)
        else:
            st.info("No per-class data for this run. Use `--layers confusion` to generate it.")

    with subtab_cm:
        if run.confusion_matrix_paths:
            st.caption(
                "Matriz de confusión normalizada: cada celda (i, j) = P(predice clase j | clase verdadera es i). "
                "La diagonal equivale al recall por clase. Las celdas fuera de la diagonal muestran confusiones entre clases."
            )
            epoch_labels = [p.stem.split("_epoch")[-1] for p in run.confusion_matrix_paths]
            cm_idx = st.selectbox(
                "Epoch", range(len(run.confusion_matrix_paths)),
                format_func=lambda i: f"Epoch {epoch_labels[i]}",
                key="cm_epoch_sel",
            )
            cm_path = run.confusion_matrix_paths[cm_idx]
            if cm_path.exists():
                st.image(Image.open(cm_path), use_container_width=True)
        else:
            st.info("No hay matriz de confusión para este run. Usa `--layers confusion` para generarla.")

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

        col_a, col_b = st.columns(2)
        with col_a:
            label_a = st.selectbox("Run A", all_labels, index=0, key="cmp_a")
        with col_b:
            label_b = st.selectbox("Run B", all_labels, index=min(1, len(all_labels) - 1), key="cmp_b")

        run_a, run_b = run_labels[label_a], run_labels[label_b]
        df_a = _load_df(str(run_a.log_path), str(run_a.epoch_csv_path) if run_a.epoch_csv_path else None)
        df_b = _load_df(str(run_b.log_path), str(run_b.epoch_csv_path) if run_b.epoch_csv_path else None)

        short_a, short_b = label_a[:25], label_b[:25]
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

# ── Tab: Feasibility ─────────────────────────────────────────────────────────

with tab_feasibility:
    feasibility_csvs = _get_feasibility_csvs()

    st.subheader("Feasibility Checker")

    # ── Selector de feasibility CSV ──────────────────────────────────────────
    if feasibility_csvs:
        csv_labels = {str(p): p.name for p in feasibility_csvs}
        selected_feas_path = st.selectbox(
            "Feasibility report", list(csv_labels.keys()),
            format_func=lambda p: csv_labels[p],
        )
        meta, bdf_feas = parse_feasibility_csv(Path(selected_feas_path))

        if meta:
            fa, fb, fc, fd = st.columns(4)
            fa.metric("Model", meta.get("model_name", "—"))
            fb.metric("Params (M)", meta.get("total_params_M", "—"))
            fc.metric("VRAM total (GB)", meta.get("total_vram_gb", "—"))
            fd.metric("Hardware", meta.get("hardware_name", "—"))

        if not bdf_feas.empty:
            st.subheader("Benchmark Results")
            st.dataframe(bdf_feas, use_container_width=True)

            viable = bdf_feas[(bdf_feas["oom"] == "no") & bdf_feas["imgs_per_s"].notna()]
            if not viable.empty:
                fig_feas = go.Figure()
                for mode in viable["trace_mode"].unique():
                    subset = viable[viable["trace_mode"] == mode]
                    fig_feas.add_trace(go.Bar(
                        x=subset["batch_size"].astype(str),
                        y=subset["imgs_per_s"],
                        name=f"trace={mode}",
                    ))
                fig_feas.update_layout(
                    barmode="group",
                    title="Throughput por batch size",
                    xaxis_title="Batch size",
                    yaxis_title="imgs/s",
                    height=350,
                )
                st.plotly_chart(fig_feas, use_container_width=True)
    else:
        st.info("No feasibility CSVs found. Run `check_feasibility.py` to generate one.")

    st.markdown("---")

    # ── Ejecutar feasibility desde la web ────────────────────────────────────
    st.subheader("Ejecutar Feasibility Check")
    with st.form("feasibility_form"):
        model_options = [
            "vit_base_patch16_224", "vit_tiny_patch16_224",
            "vit_small_patch16_224", "resnet50", "efficientnet_b0",
        ]
        feas_model = st.selectbox("Modelo", model_options)
        feas_batches = st.multiselect(
            "Batch sizes", [16, 32, 64, 128], default=[32, 64],
        )
        feas_epochs = st.number_input("Epochs para estimación", min_value=1, value=30)
        feas_traces = st.multiselect(
            "Trace modes", ["off", "simple", "deep"], default=["off", "simple"],
        )
        submitted = st.form_submit_button("Ejecutar")

    if submitted:
        if not feas_batches:
            st.error("Selecciona al menos un batch size.")
        else:
            bs_args = " ".join(str(b) for b in feas_batches)
            trace_args = " ".join(feas_traces) if feas_traces else "off"
            cmd = (
                f"uv run python scripts/check_feasibility.py "
                f"--model {feas_model} "
                f"--batch-sizes {bs_args} "
                f"--epochs {feas_epochs} "
                f"--trace-modes {trace_args}"
            )
            st.code(cmd, language="bash")
            with st.spinner("Ejecutando feasibility check…"):
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    cwd=str(ROOT),
                )
            if result.returncode == 0:
                st.success("Completado")
                st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
                # Clear cache so the new CSV appears in the selector
                _get_feasibility_csvs.clear()
            else:
                st.error("Error al ejecutar:")
                st.code(result.stderr[-2000:])

# ── Tab: Time Analysis ────────────────────────────────────────────────────────

with tab_time:
    st.subheader("Time Analysis — estimación vs real")

    if "epoch_time" not in df.columns or df["epoch_time"].isna().all():
        st.info("Sin datos de tiempo por epoch. Usa `--trace simple` para generarlos.")
    else:
        et = df[["epoch", "epoch_time"]].dropna()

        # Real epoch times
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=et["epoch"], y=et["epoch_time"] / 60,
            name="Tiempo real (min)", mode="lines+markers",
            line=dict(color="#4c72b0", width=2), marker=dict(size=5),
        ))

        # If a feasibility CSV is selected in the Feasibility tab, overlay estimate
        if feasibility_csvs:
            feas_path = feasibility_csvs[0]  # most recent
            try:
                _, bdf_t = parse_feasibility_csv(feas_path)
                best_est_col = [c for c in bdf_t.columns if c.startswith("est_min_per_epoch")]
                if not bdf_t.empty and best_est_col:
                    est_col = best_est_col[0]
                    best_viable = bdf_t[(bdf_t["oom"] == "no") & bdf_t[est_col].notna()]
                    if not best_viable.empty:
                        best_row = best_viable.loc[best_viable["imgs_per_s"].idxmax()]
                        est_min = float(best_row[est_col])
                        fig_time.add_hline(
                            y=est_min,
                            line_dash="dash", line_color="#dd8452",
                            annotation_text=f"Estimación: {est_min:.0f} min/epoch",
                            annotation_position="top left",
                        )
            except Exception:
                pass

        fig_time.update_layout(
            title="Tiempo por epoch",
            xaxis_title="Epoch", yaxis_title="Minutos",
            height=350,
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # Summary table
        total_s = et["epoch_time"].sum()
        avg_s = et["epoch_time"].mean()
        st.markdown(f"**Total real:** {int(total_s//3600)}h {int((total_s%3600)//60)}m")
        st.markdown(f"**Avg/epoch:** {avg_s/60:.1f} min")
        st.markdown(f"**Epochs:** {len(et)}")

# ── Tab: Run Info ────────────────────────────────────────────────────────────

with tab_info:
    st.subheader("Run metadata")

    info_rows = {
        "Log file": run.log_path.name,
        "Env": run.env,
        "Trace mode": run.trace_mode,
        "Epochs parsed": n_epochs,
        "Best Val F1": f"{best_f1:.4f}" if not pd.isna(best_f1) else "—",
        "Best epoch": best_epoch,
        "Epoch CSV": run.epoch_csv_path.name if run.epoch_csv_path else "—",
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
    st.markdown(f"- Per-class CSV: `{run.perclass_csv_path.name if run.perclass_csv_path else '—'}`")
    if run.perclass_paths:
        for p in run.perclass_paths:
            st.markdown(f"- Per-class PNG: `{p.name}`")

    with st.expander("View raw log (first 200 lines)"):
        lines = run.log_path.read_text(errors="replace").splitlines()[:200]
        st.code("\n".join(lines), language=None)
