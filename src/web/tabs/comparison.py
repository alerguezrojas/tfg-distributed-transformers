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
    st.markdown("## Compare")
    st.caption("Measure the real distributed speedup against single-GPU, or overlay several runs.")
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
    st.markdown("### Single vs Distributed — measured speedup")
    st.caption("Did distributing actually help? Compares a single-GPU run against any distributed "
               "strategy — DDP, heterogeneous or model-parallel — validated against the "
               "feasibility's prediction. (Predicted-only scenarios live in Feasibility.)")

    all_runs_ddp = _get_runs()
    if not all_runs_ddp:
        st.info("No runs found.")
    else:
        single_runs = [r for r in all_runs_ddp if r.mode == "single"]
        # Every distributed strategy counts: "ddp" (multi-process NCCL),
        # "ddp_hetero" (GPU+CPU) and "model_parallel" (pipeline across 2 GPUs) —
        # so 1 GPU vs model-parallel is also comparable here.
        ddp_runs = [r for r in all_runs_ddp
                    if r.mode.startswith("ddp") or r.mode == "model_parallel"]

        da1, da2, da3 = st.columns(3)
        da1.metric("Single-GPU runs", len(single_runs))
        da2.metric("Distributed runs", len(ddp_runs))
        da3.metric("Total runs", len(all_runs_ddp))

        if not ddp_runs:
            st.info("No distributed runs yet. Launch `scripts/train_ddp.py` or "
                    "`scripts/train_model_parallel.py` — results will appear here automatically.")
        else:
            st.markdown("### Distributed runs")
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
            with col_d:
                ddp_lbl = st.selectbox("Distributed run", [r.label for r in ddp_runs], key="ddp_ddp_sel")
            r_ddp = next(r for r in ddp_runs if r.label == ddp_lbl)

            # Sensible default for the single run: same model, environment and
            # precision as the chosen distributed run, and not deep-trace (deep
            # adds ~20% overhead, which would inflate the measured speedup).
            # Runs without a precision marker predate the selector → fp32.
            def _prec(r):
                return r.precision or "fp32"

            def _single_rank(r):
                return (
                    r.model == r_ddp.model,
                    r.env == r_ddp.env,
                    _prec(r) == _prec(r_ddp),
                    r.trace_mode != "deep",
                    r.sort_key,
                )
            _default_single = max(single_runs, key=_single_rank)
            _single_idx = single_runs.index(_default_single)
            with col_s:
                single_lbl = st.selectbox(
                    "Single-GPU run", [r.label for r in single_runs],
                    index=_single_idx, key="ddp_single_sel",
                )
            r_single = next(r for r in single_runs if r.label == single_lbl)

            # Comparability warnings: mismatches silently distort the speedup.
            if r_single.model != r_ddp.model:
                st.warning(f"The runs use **different models** ({r_single.model} vs {r_ddp.model}) "
                           "— the speedup is not apples-to-apples.")
            if _prec(r_single) != _prec(r_ddp):
                st.warning(f"The runs use **different precision** ({_prec(r_single)} vs "
                           f"{_prec(r_ddp)}) — Tensor cores alone change the time ~3-4×, so the "
                           "speedup mixes two effects. Prefer same-precision runs.")
            if r_single.trace_mode == "deep" and r_ddp.trace_mode != "deep":
                st.warning("The single-GPU run uses **deep tracing** (~20% overhead) — "
                           "the measured speedup is inflated. Prefer a simple-trace single run.")

            df_s = _load_df(str(r_single.log_path), str(r_single.epoch_csv_path) if r_single.epoch_csv_path else None)
            df_d = _load_df(str(r_ddp.log_path), str(r_ddp.epoch_csv_path) if r_ddp.epoch_csv_path else None)

            if not df_s.empty and not df_d.empty:
                avg_s = df_s["epoch_time"].mean() if "epoch_time" in df_s.columns and df_s["epoch_time"].notna().any() else None
                avg_d = df_d["epoch_time"].mean() if "epoch_time" in df_d.columns and df_d["epoch_time"].notna().any() else None

                # Correct labels by distribution type: the heterogeneous case is
                # V100+CPU (NOT 2 equivalent GPUs), and model parallelism splits
                # the MODEL across 2 GPUs (data parallelism's "ideal 2x" does not
                # apply — naive pipeline stages serialize, so ~1x is expected).
                is_hetero = r_ddp.mode == "ddp_hetero"
                is_mp = r_ddp.mode == "model_parallel"
                if is_hetero:
                    worker_desc, dist_label = "V100 + CPU", "Hetero (V100+CPU)"
                elif is_mp:
                    worker_desc, dist_label = "2 GPUs, pipeline", "Model-parallel"
                else:
                    worker_desc, dist_label = "2 GPUs", "DDP"
                n_workers = 2

                su1, su2, su3, su4 = st.columns(4)
                su1.metric("Single-GPU epoch", f"{avg_s/60:.1f} min" if avg_s else "—")
                su2.metric(f"Distributed epoch ({worker_desc})", f"{avg_d/60:.1f} min" if avg_d else "—")
                speedup = None
                if avg_s and avg_d and avg_d > 0:
                    speedup = avg_s / avg_d
                    su3.metric("Real speedup", f"{speedup:.2f}×")
                    if is_mp:
                        su4.metric("Expected (naive pipeline)", "≈1×")
                    else:
                        su4.metric(f"Efficiency vs ideal {n_workers}× ", f"{speedup / n_workers * 100:.1f}%")

                if speedup is not None and is_mp:
                    st.info(
                        f"**Model parallelism: {speedup:.2f}× — it does not accelerate, and that is "
                        f"the expected result.** The naive pipeline serializes the stages (while one "
                        f"GPU computes, the other waits), so ≈1× is the theoretical ceiling. Its value "
                        f"is **fitting models that do not fit on one GPU**: vit_large OOMs on a single "
                        f"T4 but trains split 12/24 across both."
                    )
                elif speedup is not None and speedup < 1:
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
                # here so the whole distributed story lives in one place. (The
                # DDPOptimizer predicts DATA parallelism — not applicable to
                # heterogeneous or model-parallel runs.)
                if speedup is not None and not is_hetero and not is_mp:
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
                                                     name=f"{dist_label} Val F1", line=dict(color=COLORS[2], width=2)))
                fig_ddp_f1.update_layout(**_base_layout(300, f"Val F1: Single-GPU vs {dist_label}"),
                                          xaxis_title="Epoch", yaxis_title="Val F1")
                _show(fig_ddp_f1, "ddp_f1")

                if avg_s and avg_d:
                    fig_time_ddp = go.Figure()
                    if "epoch_time" in df_s.columns:
                        fig_time_ddp.add_trace(go.Scatter(x=df_s["epoch"], y=df_s["epoch_time"] / 60,
                                                           name="Single-GPU", line=dict(color=COLORS[0], width=2)))
                    if "epoch_time" in df_d.columns:
                        fig_time_ddp.add_trace(go.Scatter(x=df_d["epoch"], y=df_d["epoch_time"] / 60,
                                                           name=dist_label, line=dict(color=COLORS[2], width=2)))
                    fig_time_ddp.update_layout(**_base_layout(260, f"Epoch time: Single-GPU vs {dist_label}"),
                                               xaxis_title="Epoch", yaxis_title="Minutes")
                    _show(fig_time_ddp, "ddp_time")

                # Worker-scaling theory describes DATA parallelism (N workers on
                # the same data) — it is meaningless for a pipeline that splits
                # the model, so the section is skipped for model-parallel runs.
                if not is_mp:
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
            "Select runs to compare (max 8)", all_labels_list,
            default=all_labels_list[:min(2, len(all_labels_list))],
            max_selections=8,
        )

        if len(selected_compare) < 2:
            st.info("Select at least 2 runs.")
        else:
            compare_runs_list = [(lbl, all_run_labels[lbl]) for lbl in selected_compare]
            compare_dfs: list[tuple[str, pd.DataFrame]] = []
            for lbl, r in compare_runs_list:
                cdf = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
                # The log reports train energy in Joules — derive Wh so energy
                # charts use one unit (eval already comes as energy_eval_wh).
                if "energy_train_j" in cdf.columns:
                    cdf = cdf.assign(energy_train_wh=cdf["energy_train_j"] / 3600.0)
                compare_dfs.append((lbl, cdf))

            summary_rows = []
            for lbl, r in compare_runs_list:
                cdf = next(d for ll, d in compare_dfs if ll == lbl)
                best_f1_c = _safe_max(cdf["val_f1"]) if "val_f1" in cdf.columns and not cdf.empty else float("nan")
                best_ep_c_v = _safe_val_at_best(cdf, "val_f1", "epoch")
                _last = cdf["val_f1"].dropna() if "val_f1" in cdf.columns and not cdf.empty else pd.Series(dtype=float)
                total_s_c = cdf["epoch_time"].dropna().sum() if "epoch_time" in cdf.columns else float("nan")
                summary_rows.append({
                    "Run": lbl,
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
                    fill="toself", name=lbl,
                    line=dict(color=COLORS[i % len(COLORS)]), opacity=0.6,
                ))
            # Full-label legend below the radar: one row per run stays readable
            # even with 7-8 runs selected.
            _n_radar = len(compare_dfs)
            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, height=380 + 20 * _n_radar,
                legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
                margin=dict(l=60, r=60, t=40, b=40 + 20 * _n_radar), paper_bgcolor="white",
                title=dict(text="Metrics at the best Val F1 epoch", font=dict(size=13)),
            )
            _show(radar_fig, "radar_comparison")
            st.markdown("---")

            # ── Energy comparison (runs trained with --fn energy) ─────────────
            def _has(d: pd.DataFrame, c: str) -> bool:
                return c in d.columns and d[c].notna().any()

            energy_rows = []
            for lbl, cdf in compare_dfs:
                t_wh = cdf["energy_train_wh"].dropna().sum() if _has(cdf, "energy_train_wh") else 0.0
                e_wh = cdf["energy_eval_wh"].dropna().sum() if _has(cdf, "energy_eval_wh") else 0.0
                if t_wh or e_wh:
                    energy_rows.append((lbl, t_wh, e_wh))

            if energy_rows:
                st.markdown("#### Energy consumption")
                st.caption(
                    "Total energy over the whole run (Wh), as measured by pynvml on the "
                    "logging GPU. Runs without energy measurement (no `--fn energy`, e.g. "
                    "model-parallel) are not shown."
                )
                fig_energy = go.Figure()
                _lbls = [l for l, _, _ in energy_rows]
                fig_energy.add_trace(go.Bar(
                    y=_lbls, x=[t for _, t, _ in energy_rows], name="Train",
                    orientation="h", marker_color=COLORS[0],
                ))
                fig_energy.add_trace(go.Bar(
                    y=_lbls, x=[e for _, _, e in energy_rows], name="Eval",
                    orientation="h", marker_color=COLORS[1],
                ))
                fig_energy.update_layout(
                    **_base_layout(160 + 44 * len(energy_rows), "Total energy per run (Wh)",
                                   margin=dict(l=10, r=16, t=48, b=40)),
                    barmode="stack", xaxis_title="Wh",
                )
                fig_energy.update_yaxes(autorange="reversed", automargin=True)
                # Outside the plot: the inside-top-left default would cover the first bar.
                fig_energy.update_layout(legend=dict(
                    orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)",
                ))
                _show(fig_energy, "compare_energy_total")

                _n_eff = [(l, (t + e)) for l, t, e in energy_rows]
                _best = min(_n_eff, key=lambda x: x[1])
                _worst = max(_n_eff, key=lambda x: x[1])
                if _worst[1] > 0 and _best != _worst:
                    st.caption(
                        f"Most efficient: **{_best[0]}** ({_best[1]:.1f} Wh) — "
                        f"{_worst[1]/_best[1]:.1f}× less energy than **{_worst[0]}** "
                        f"({_worst[1]:.1f} Wh)."
                    )
                st.markdown("---")

            _energy_opts = (["energy_train_wh", "energy_eval_wh", "power_train_w"]
                            if energy_rows else [])
            metrics_to_compare = st.multiselect(
                "Metrics to overlay",
                ["val_f1", "val_loss", "train_f1", "train_loss", "val_prec", "val_rec",
                 "epoch_time"] + _energy_opts,
                default=["val_f1", "val_loss"],
            )
            cols = st.columns(2)
            for idx, col_name in enumerate(metrics_to_compare):
                fig = _overlay_fig(compare_dfs, col=col_name,
                                   title=col_name.replace("_", " "), y_label=col_name)
                with cols[idx % 2]:
                    _show(fig, f"compare_{col_name}")

