"""Run results — batch view."""
from __future__ import annotations


import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (_load_batch)


def _batch(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    elif not run.batch_csv_path:
        st.info(
            "No batch-level CSV for this run. "
            "Use `--layers batch-monitor` to generate it. "
            "With `--batch-log-every 1` you get one record per individual batch."
        )
    else:
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

            st.caption(
                "Training feeds the images in small **batches** (an *epoch* = all "
                "batches once). The **Curves** tab plots one point per epoch; this tab "
                "plots one point per **batch** — much finer — to spot loss spikes / "
                "instability and to check the learning-rate schedule."
            )

            # Quick summary
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Epochs recorded", len(epochs_available_b))
            bc2.metric("Batches per epoch", n_batches_total)
            bc3.metric("Total records", len(bdf))
            if has_lr:
                bc4.metric("Initial LR", f"{bdf['lr'].iloc[0]:.2e}")

            # Single-level view switcher (replaces the old nested tab row). Default is
            # the whole-training curve — the most intuitive of the three.
            view = st.radio("View", ["Whole training", "Compare epochs", "Learning rate"],
                            horizontal=True, key="batch_view")

            # ── View: compare epochs ───────────────────────────────────────────
            if view == "Compare epochs":
                st.caption("Each selected epoch as its own line, x = batch number "
                           "*within* the epoch — to compare the shape of one epoch vs "
                           "another (e.g. does the loss settle as training advances?). "
                           "Red ✕ = loss spikes.")
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
                        help="Running mean loss = smoothed average so far · "
                             "Instantaneous batch loss = the loss of that single batch "
                             "(noisier).",
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

            # ── View: whole training (x axis = global batch) ───────────────────
            elif view == "Whole training":
                st.caption("The entire training as one line, x = batch number from "
                           "start to finish (all epochs in a row). Dotted vertical "
                           "lines mark where each epoch ends. The natural training "
                           "curve, at batch resolution.")
                col_gm, col_gma = st.columns([2, 2])
                with col_gm:
                    global_metric = st.selectbox(
                        "Global metric", _available_batch_metrics(),
                        format_func=lambda m: _BATCH_METRIC_LABELS.get(m, m),
                        key="global_metric_sel",
                        help="Running mean loss = smoothed average so far · "
                             "Instantaneous batch loss = the loss of that single batch "
                             "(noisier).",
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

            # ── View: learning rate ────────────────────────────────────────────
            elif view == "Learning rate":
                if not has_lr:
                    st.info(
                        "No learning-rate data. "
                        "Requires `--layers batch-monitor` with the current version of BatchMonitorDecorator."
                    )
                else:
                    st.caption("The learning rate over training: the linear **warmup** "
                               "(ramp up) followed by the **cosine** decay (ramp down). "
                               "Confirms the schedule ran as configured.")
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



