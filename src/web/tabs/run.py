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

from src.web.confusion_matrix_parser import (
    get_matrix_for_epoch, parse_confusion_matrix_csv,
    recall_by_class, top_confusions, confusion_profile,
)
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
    _feas_label, _run_config, _load_class_distribution, _load_example_images,
    _safe_max, _safe_idxmax, _safe_val_at_best, _throughput_col, _dur_str,
    _get_configs, _detect_anomalies, _read_log_tail, _parse_log_progress,
    _gpu_usage, _launch_process, _color_f1_cell,
)


def render(ctx: DashboardContext) -> None:
    st.markdown("## Run results")
    st.caption("Metrics and metadata of the run selected in the sidebar.")
    # One level of tabs only — the per-class trend and the confusion views used
    # to be a second nested row; they are now top-level tabs.
    sub = st.tabs(["Curves", "Per-class", "Confusions", "Batch", "Details"])
    with sub[0]:
        _curves(ctx)
    with sub[1]:
        _per_class(ctx)
    with sub[2]:
        _confusions_tab(ctx)
    with sub[3]:
        _batch(ctx)
    with sub[4]:
        # "Details" merges timing and metadata (config, anomalies, log).
        _time(ctx)
        st.markdown("---")
        _info(ctx)


def _curves(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
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

            # One-line verdict: best epoch, overfitting gap, val-loss divergence.
            if not pd.isna(best_f1) and best_ep_v is not None and "train_f1" in df.columns:
                _tr = df.loc[df["epoch"] == best_ep_v, "train_f1"]
                bits = [f"Best Val F1 **{best_f1:.3f}** at epoch {int(best_ep_v)}"]
                if not _tr.empty:
                    gap = float(_tr.iloc[0]) - float(best_f1)
                    bits.append(f"train–val gap {gap:+.2f}" + (" → overfitting" if gap > 0.1 else ""))
                if "val_loss" in df.columns and df["val_loss"].notna().sum() > 2:
                    vl = df.sort_values("epoch")["val_loss"].dropna()
                    if len(vl) > 2 and vl.iloc[-1] > vl.min() * 1.15:
                        bits.append("val loss diverges after the best epoch")
                (st.warning if "overfitting" in " ".join(bits) else st.info)(" · ".join(bits))

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



def _per_class(ctx: DashboardContext) -> None:
    """Per-class metrics at one epoch (bars + table) and their trend across
    epochs — on a single page, no nested tabs."""
    selected_run = ctx.selected_run
    run = ctx.run
    if selected_run is None:
        st.info("Select a run in the sidebar.")
        return
    if not (run.perclass_csv_path and run.perclass_csv_path.exists()):
        st.info("No per-class data. Use `--layers confusion` to generate it.")
        return

    pcdf = _load_perclass(str(run.perclass_csv_path))
    epochs_available = sorted(pcdf["epoch"].unique().tolist())
    selected_ep = st.selectbox("Epoch", epochs_available, index=len(epochs_available) - 1,
                               format_func=lambda e: f"Epoch {e}")
    ep_df = pcdf[pcdf["epoch"] == selected_ep].copy().sort_values("f1", ascending=False)

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

    with st.expander("Per-class table"):
        styled = (
            ep_df[["class_name", "f1", "precision", "recall"]]
            .style.map(_color_f1_cell, subset=["f1"])
            .format({"f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}"})
        )
        st.dataframe(styled, use_container_width=True, height=280)
        _dl_csv(ep_df[["class_name", "f1", "precision", "recall"]],
                f"perclass_ep{selected_ep}.csv", "Download per-class table")

    st.markdown("#### Trend across epochs")
    classes = sorted(pcdf["class_name"].unique().tolist())
    col_sel, col_met = st.columns([3, 1])
    with col_sel:
        selected_classes = st.multiselect("Classes (max 8)", classes,
                                           default=classes[:4], max_selections=8)
    with col_met:
        metric_sel = st.radio("Metric", ["f1", "precision", "recall"])
    if selected_classes:
        fig_trend = go.Figure()
        for i, cls in enumerate(selected_classes):
            cdf = pcdf[pcdf["class_name"] == cls].sort_values("epoch")
            fig_trend.add_trace(go.Scatter(
                x=cdf["epoch"], y=cdf[metric_sel], name=cls, mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2), marker=dict(size=4),
            ))
        fig_trend.update_layout(
            **_base_layout(400, f"{metric_sel.capitalize()} per class across epochs"),
            xaxis_title="Epoch",
        )
        fig_trend.update_yaxes(range=[0, 1])
        _show(fig_trend, "class_trend")


def _confusions_tab(ctx: DashboardContext) -> None:
    if ctx.selected_run is None:
        st.info("Select a run in the sidebar.")
        return
    _confusions_view(ctx.run)


def _confusions_view(run) -> None:
    """Multi-label confusion diagnostics.

    This is a multi-label task, so a classic N×N confusion matrix does not apply.
    Instead we show the three things that ARE interpretable from the stored
    co-activation matrix: recall per class (did the model catch the class), the
    strongest label confusions, and a per-class 'what else fires' profile. The
    full 19×19 matrix stays available as an advanced view.
    """
    if not (run.confusion_matrix_csv_path and run.confusion_matrix_csv_path.exists()):
        st.info("No confusion data. Use `--layers confusion` to generate it.")
        return

    cm_df = parse_confusion_matrix_csv(run.confusion_matrix_csv_path)
    epochs_cm = sorted(cm_df["epoch"].unique().tolist())
    ep = st.selectbox("Epoch", epochs_cm, index=len(epochs_cm) - 1,
                      format_func=lambda e: f"Epoch {e}", key="cm_epoch_sel")

    st.caption(
        "Multi-label task: each image can carry several of the 19 classes, so there "
        "is no single 'predicted class' to confuse. These views read the model's "
        "label co-activation: **recall** (whether each class is detected) and which "
        "other labels are predicted when a class is present (confusion / co-occurrence)."
    )

    # ── 1) Recall per class (the diagonal) ────────────────────────────────────
    st.markdown("#### Recall by class")
    rec = recall_by_class(cm_df, ep)
    if not rec.empty:
        bar_colors = [
            COLORS[3] if v < 0.3 else (COLORS[1] if v < 0.6 else COLORS[2])
            for v in rec.values
        ]
        fig_rec = go.Figure(go.Bar(
            y=list(rec.index), x=list(rec.values), orientation="h",
            marker_color=bar_colors, text=[f"{v:.2f}" for v in rec.values],
            textposition="outside",
        ))
        fig_rec.update_layout(
            **_base_layout(120 + 26 * len(rec), "Recall by class (red < 0.30, amber < 0.60, green ≥ 0.60)",
                           margin=dict(l=200, r=40, t=48, b=40)),
            showlegend=False,
        )
        fig_rec.update_xaxes(range=[0, 1], title="Recall")
        fig_rec.update_yaxes(automargin=True)
        _show(fig_rec, f"recall_by_class_ep{ep}")
        failed = rec[rec < 0.3]
        if not failed.empty:
            st.warning(
                f"**{len(failed)} class(es) the model rarely catches** (recall < 0.30): "
                + ", ".join(f"{c} ({v:.2f})" for c, v in failed.items())
                + ". Typically rare classes, which lower the macro-F1."
            )

    st.markdown("---")

    # ── 2) Strongest confusions (off-diagonal) ────────────────────────────────
    st.markdown("#### Label confusions")
    st.caption("When the class on the left is truly present, the model also predicts "
               "the class on the right this often. Some pairs are real confusion, "
               "others are natural co-occurrence (e.g. forest types share scenes).")
    top = top_confusions(cm_df, ep, k=12)
    if top.empty:
        st.info("No strong off-diagonal confusions at this epoch.")
    else:
        pair_labels = [f"{r.true_class}  →  {r.pred_class}" for r in top.itertuples()]
        fig_top = go.Figure(go.Bar(
            y=pair_labels, x=list(top["value"]), orientation="h",
            marker_color=COLORS[0], text=[f"{v:.2f}" for v in top["value"]],
            textposition="outside",
        ))
        fig_top.update_layout(
            **_base_layout(120 + 28 * len(top), "Top confusions (true → also predicted)",
                           margin=dict(l=300, r=40, t=48, b=40)),
            showlegend=False,
        )
        fig_top.update_xaxes(range=[0, 1], title="P(also predicts the right label | left is present)")
        fig_top.update_yaxes(automargin=True, autorange="reversed")
        _show(fig_top, f"top_confusions_ep{ep}")
        _dl_csv(top, f"top_confusions_ep{ep}.csv", "Download confusions table")

    st.markdown("---")

    # ── 3) Per-class confusion profile ────────────────────────────────────────
    st.markdown("#### Per-class profile")
    all_classes = sorted(cm_df["true_class"].unique().tolist())
    sel_cls = st.selectbox("When this class is truly present…", all_classes, key="cm_profile_cls")
    prof = confusion_profile(cm_df, ep, sel_cls).head(10)
    diag = recall_by_class(cm_df, ep).get(sel_cls, float("nan"))
    st.caption(f"Recall of **{sel_cls}**: {diag:.2f} — detected in {diag*100:.0f}% of "
               f"cases. Below: the other labels also predicted when it is present.")
    if not prof.empty and prof.max() > 0:
        fig_prof = go.Figure(go.Bar(
            y=list(prof.index), x=list(prof.values), orientation="h",
            marker_color=COLORS[5], text=[f"{v:.2f}" for v in prof.values],
            textposition="outside",
        ))
        fig_prof.update_layout(
            **_base_layout(120 + 26 * len(prof), f"Also predicted when '{sel_cls}' is present",
                           margin=dict(l=200, r=40, t=48, b=40)),
            showlegend=False,
        )
        fig_prof.update_xaxes(range=[0, 1], title="Frequency")
        fig_prof.update_yaxes(automargin=True, autorange="reversed")
        _show(fig_prof, f"profile_{ep}")
    else:
        st.info("The model rarely turns on other labels for this class.")

    # ── 4) Full matrix (advanced) ─────────────────────────────────────────────
    with st.expander("Full 19×19 co-activation matrix (advanced)"):
        st.caption("Cell (row i, column j) = P(model predicts j | class i is truly "
                   "present). The diagonal is recall; bright off-diagonal cells are "
                   "the confusions above. Colored borders group classes by ecosystem.")
        pivot = get_matrix_for_epoch(cm_df, ep)
        class_order = list(pivot.index)
        z_norm = pivot.reindex(index=class_order, columns=class_order).values
        n_classes = len(class_order)
        text = [[f"{v:.2f}" if v >= 0.05 else "" for v in row] for row in z_norm]

        shapes = []
        for _gname, (idxs, color) in _CLASS_GROUPS.items():
            positions = [i for i in range(n_classes) if i in idxs]
            if not positions:
                continue
            lo, hi = min(positions), max(positions)
            shapes.append(dict(
                type="rect", x0=lo - 0.5, x1=hi + 0.5, y0=lo - 0.5, y1=hi + 0.5,
                line=dict(color=color, width=2.5), fillcolor="rgba(0,0,0,0)", layer="above",
            ))

        fig_cm = go.Figure(go.Heatmap(
            z=z_norm.tolist(), x=class_order, y=class_order,
            colorscale="Blues", zmin=0, zmax=1,
            text=text, texttemplate="%{text}", textfont={"size": 8},
            hovertemplate="True: %{y}<br>Also predicts: %{x}<br>P = %{z:.3f}<extra></extra>",
            colorbar=dict(title="P(pred j | true i)"),
        ))
        fig_cm.update_layout(
            title=dict(text=f"Co-activation matrix — Epoch {ep}", font=dict(size=13)),
            xaxis=dict(title="Also predicted", tickangle=45, tickfont=dict(size=9),
                       tickmode="array", tickvals=list(range(n_classes)), ticktext=class_order),
            yaxis=dict(title="True class", tickfont=dict(size=9), autorange="reversed",
                       tickmode="array", tickvals=list(range(n_classes)), ticktext=class_order),
            height=660, margin=dict(l=180, r=20, t=50, b=180),
            paper_bgcolor="white", shapes=shapes,
        )
        _show(fig_cm, f"confusion_matrix_ep{ep}")



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

            # Quick summary
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Epochs recorded", len(epochs_available_b))
            bc2.metric("Batches per epoch", n_batches_total)
            bc3.metric("Total records", len(bdf))
            if has_lr:
                bc4.metric("Initial LR", f"{bdf['lr'].iloc[0]:.2e}")

            # Single-level view switcher (replaces the old nested tab row).
            view = st.radio("View", ["Per epoch", "Global history", "Learning rate"],
                            horizontal=True, key="batch_view")

            # ── View: per epoch ────────────────────────────────────────────────
            if view == "Per epoch":
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

            # ── View: global history (x axis = global batch) ───────────────────
            elif view == "Global history":
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

            # ── View: learning rate ────────────────────────────────────────────
            elif view == "Learning rate":
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



def _time(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
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



def _info(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
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

