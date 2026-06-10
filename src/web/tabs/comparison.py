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
    _feas_label, _run_config, _load_class_distribution, _load_example_images,
    _safe_max, _safe_idxmax, _safe_val_at_best, _throughput_col, _dur_str,
    _get_configs, _detect_anomalies, _read_log_tail, _parse_log_progress,
    _gpu_usage, _launch_process, _color_f1_cell,
)


def _predicted_2gpu_speedup(env: str, model: str) -> float | None:
    """The feasibility's predicted 2-GPU speedup for this env/model, if any."""
    for p in _get_feasibility_csvs():
        try:
            if p.parent.parent.name != env:
                continue
            meta, _ = parse_feasibility_csv(p)
            if meta.get("model_name") != model:
                continue
            scen = parse_ddp_scenarios(meta)
            if not scen.empty and {"n_gpus", "speedup"}.issubset(scen.columns):
                r2 = scen[scen["n_gpus"] == 2]
                if not r2.empty:
                    return float(r2.iloc[0]["speedup"])
        except Exception:
            pass
    return None


def render(ctx: DashboardContext) -> None:
    sub = st.tabs(["Single vs Distributed", "Overlay runs"])
    with sub[0]:
        _ddp(ctx)
    with sub[1]:
        _overlay(ctx)


def _ddp(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    st.markdown("## Single vs Distributed — measured speedup")
    st.caption("Did distributing actually help? Measured here, and validated against "
               "the feasibility's prediction. (Predicted-only scenarios live in Feasibility.)")
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

                # Predicted vs measured — bring the feasibility's 2-GPU prediction
                # here so the whole distributed story lives in one place.
                if speedup is not None and not is_hetero:
                    pred_sp = _predicted_2gpu_speedup(r_ddp.env, r_ddp.model)
                    if pred_sp:
                        err = (pred_sp - speedup) / speedup * 100
                        pp1, pp2, pp3 = st.columns(3)
                        pp1.metric("Predicted speedup (feasibility)", f"{pred_sp:.2f}×")
                        pp2.metric("Measured speedup", f"{speedup:.2f}×")
                        pp3.metric("Prediction error", f"{err:+.0f}%")
                        ok = abs(err) <= 15
                        (st.success if ok else st.info)(
                            f"The feasibility predicts the speedup from a **1-GPU** benchmark; "
                            f"here it is validated against the real multi-GPU run "
                            f"({'accurate' if ok else 'off'}: predicted {pred_sp:.2f}× vs measured {speedup:.2f}×)."
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



def _overlay(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
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

