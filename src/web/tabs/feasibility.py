"""Tab render module — see src/web/app.py for the orchestrator."""
from __future__ import annotations

import shlex
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
from src.performance_model import (
    GPU_TABLE, MODEL_TABLE, predict, predict_quality, estimate_rc,
    gpu_spec, model_spec,
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
    _gpu_usage, _color_f1_cell,
)


def render(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    st.markdown("## Feasibility")
    st.caption("Plan before training. **Predict** with the analytic model (no run "
               "needed), **Validate** predictions against real trainings, or "
               "**Measure** on this machine to calibrate.")
    # Three clear stages instead of five overlapping tabs:
    #   Predict  → the analytic predictor (formulas, no benchmark) — the main tool
    #   Validate → predicted vs actual (evidence the formulas work)
    #   Measure  → run the real benchmark on this machine (advanced/calibration):
    #              generate a report, view it, run the convergence study
    tab_predict, tab_validate, tab_measure = st.tabs(
        ["Predict", "Validate", "Measure (advanced)"]
    )

    with tab_predict:
        _analytic_predictor()

    # Validate = the old "Prediction vs reality" block fills this tab.
    subtab_predreal = tab_validate

    # Measure = a single scrolling page (no nested tab row): the three blocks
    # below fill these containers in order — generate, view report, study.
    with tab_measure:
        st.caption("Run the real benchmark on the machine you are on. Use it to "
                   "calibrate the predictor or to profile this hardware.")
        st.markdown("#### Generate a report")
        subtab_run_feas = st.container()
        st.markdown("#### Report")
        subtab_report = st.container()
        st.markdown("#### Convergence study")
        subtab_study = st.container()

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
        st.markdown("### Predicted vs actual")
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
                                st.warning(f"**Optimistic estimate** — the run was {abs(_e):.0f}% slower than predicted.{_x}")
                            else:
                                st.info(f"**Conservative estimate** — the run was {_e:.0f}% faster than predicted.")
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
                                                  marker_color="#3A536B",
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
            subtab_ddp_opt = st.container()
        else:
            # One scrolling page with collapsible sections (no nested tab row).
            # The first is open; the rest start collapsed → summary-first.
            _rt_titles = [
                "Hardware & precision", "Dataset I/O & memory",
                "Throughput & time", "Distributed scaling", "Cloud cost",
            ]
            _rt = [st.expander(t, expanded=(i == 0)) for i, t in enumerate(_rt_titles)]

            with _rt[0]:
                # ── System profile ─────────────────────────────────────────────
                st.markdown("### System profile")
                hw_col1, hw_col2, hw_col3, hw_col4 = st.columns(4)
                hw_col1.metric("Model", meta.get("model_name", "—"))
                hw_col2.metric("Parameters (M)", meta.get("total_params_M", "—"))
                hw_col3.metric("GPU", meta.get("hardware_name", "—"))
                hw_col4.metric("Total VRAM (GB)", meta.get("total_vram_gb", "—"))

                # GPU specs (compute capability × SM count) if available
                gpu = meta.get("gpu", {})
                if gpu and gpu.get("cuda_cores"):
                    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                    gc1.metric("Architecture", gpu.get("architecture", "—"))
                    gc2.metric("Compute cap.", gpu.get("compute_capability", "—"))
                    gc3.metric("SMs", gpu.get("sm_count", "—"))
                    gc4.metric("CUDA cores", f"{int(gpu['cuda_cores']):,}")
                    tc = gpu.get("tensor_cores", 0)
                    gc5.metric("Tensor cores", f"{int(tc):,}" if tc else "0")

                # Precision (Tensor-core) comparison if measured
                pc = meta.get("precision_cmp")
                if pc and pc.get("fp32_imgs_s"):
                    st.markdown("#### Precision — CUDA cores vs Tensor cores")
                    st.caption(
                        "You don't pick cores directly: choosing the **numeric precision** decides "
                        "which units do the matrix math. **FP32** runs on the conventional CUDA cores; "
                        "**TF32/FP16/BF16** route the heavy matmuls through the **Tensor cores** "
                        "(faster and less VRAM, with negligible accuracy change)."
                    )
                    pcl, pcr = st.columns([2, 3])
                    with pcl:
                        p1, p2 = st.columns(2)
                        p1.metric("FP32 (CUDA cores)", f"{pc['fp32_imgs_s']:.0f} img/s")
                        p2.metric(f"{str(pc.get('tc_precision','amp')).upper()} (Tensor cores)",
                                  f"{pc['tc_imgs_s']:.0f} img/s",
                                  delta=f"{pc['speedup']:.2f}× faster")
                        if pc.get("fp32_vram_gb") and pc.get("tc_vram_gb"):
                            st.caption(f"VRAM: {pc['fp32_vram_gb']:.2f} → {pc['tc_vram_gb']:.2f} GB "
                                       f"(batch {int(pc.get('batch_size', 0))})")
                    with pcr:
                        fig_pc = go.Figure(go.Bar(
                            x=["FP32 (CUDA cores)", f"{str(pc.get('tc_precision','amp')).upper()} (Tensor cores)"],
                            y=[pc["fp32_imgs_s"], pc["tc_imgs_s"]],
                            marker_color=[COLORS[5], COLORS[2]],
                            text=[f"{pc['fp32_imgs_s']:.0f}", f"{pc['tc_imgs_s']:.0f}"],
                            textposition="outside",
                        ))
                        fig_pc.update_layout(**_base_layout(220, "Training throughput by precision"),
                                             yaxis_title="img/s")
                        _show(fig_pc, "precision_cmp")
                elif meta.get("precision") and meta.get("precision") != "fp32":
                    st.caption(f"Benchmark precision: **{meta['precision']}** (Tensor cores).")

                # CPU if available
                cpu = meta.get("cpu", {})
                if cpu:
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Logical cores", cpu.get("logical_cores", "—"))
                    cc2.metric("Physical cores", cpu.get("physical_cores", "—"))
                    cc3.metric("Total RAM (GB)", cpu.get("ram_total_gb", "—"))
                    cc4.metric("Free RAM (GB)", cpu.get("ram_free_gb", "—"))

            with _rt[1]:
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
                else:
                    st.caption("No dataset-I/O profile in this report (run with a `--dataset-path`).")

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

            with _rt[2]:
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

            with _rt[3]:
                # DDP scenarios (1/2/4/8 GPUs) — filled by the DDP analysis block below.
                subtab_ddp_opt = st.container()

            with _rt[4]:
                # ── Cloud training cost (estimated) ────────────────────────────
                st.markdown("### Cloud training cost (estimated)")
                from src.cloud_cost import estimate_costs
                viable_c = bdf_feas[bdf_feas["oom"] == "no"].copy() if not bdf_feas.empty else bdf_feas
                per_ep_col = next((c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                                   if not bdf_feas.empty and c in bdf_feas.columns), None)
                if per_ep_col is None or viable_c.empty:
                    st.info("No time estimate in this report — run a benchmark first.")
                else:
                    _orig = next((c for c in bdf_feas.columns
                                  if c.startswith("est_total_h_") and c.endswith("ep")), None)
                    _def_ep = 30
                    if _orig:
                        try:
                            _def_ep = int(_orig.split("est_total_h_")[1].replace("ep", ""))
                        except ValueError:
                            pass
                    cc_ep = st.number_input("Epochs", min_value=1, value=_def_ep, key="cloud_cost_epochs")
                    best_min = float(viable_c[per_ep_col].min())   # fastest viable config
                    total_h = best_min * cc_ep / 60.0
                    ref_gpu = meta.get("hardware_name", "")
                    st.caption(
                        f"Based on **{total_h:.1f} h** total on the benchmarked GPU "
                        f"({ref_gpu or 'unknown'}) for {cc_ep} epochs. Times are scaled to each GPU by "
                        "relative FP16 throughput; prices are approximate on-demand rates "
                        "(editable in `src/cloud_cost.py`), so treat this as a ballpark."
                    )
                    rows = estimate_costs(total_h, ref_gpu)
                    cost_df = pd.DataFrame([{
                        "Provider": r["provider"], "GPU": r["gpu"],
                        "$/h": r["usd_per_hour"], "Est. hours": r["est_hours"],
                        "Est. cost ($)": r["cost_usd"], "Notes": r["note"],
                    } for r in rows])
                    st.dataframe(cost_df, use_container_width=True, hide_index=True, height=320)
                    _dl_csv(cost_df, "cloud_cost.csv", "Download cost table")
                    paid = [r for r in rows if r["cost_usd"] > 0]
                    if paid:
                        fig_cost = go.Figure(go.Bar(
                            x=[f"{r['provider']} · {r['gpu']}" for r in paid],
                            y=[r["cost_usd"] for r in paid],
                            marker_color=COLORS[0], opacity=0.85,
                            text=[f"${r['cost_usd']:.0f}" for r in paid], textposition="outside",
                        ))
                        fig_cost.update_layout(**_base_layout(320, "Estimated training cost by provider (paid)"),
                                               xaxis_tickangle=35, yaxis_title="USD")
                        _show(fig_cost, "cloud_cost")
                        st.caption("Free options (Kaggle / Colab) are in the table above, not the chart.")

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
            st.markdown("## Distributed scaling (predicted)")
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

                    # ── Speedup: feasibility estimate vs selectable scaling laws ──
                    st.markdown("### Speedup: estimate vs scaling laws")
                    if "speedup" in viable_ddp.columns:
                        from src.estimation_models import SPEEDUP_MODELS, speedup_curve
                        n_gpus_vals = viable_ddp["n_gpus"].tolist()
                        speedup_vals = viable_ddp["speedup"].tolist()

                        st.caption(
                            "The green line is the feasibility's own **compute/IO-aware** estimate "
                            "(includes NFS I/O and gradient-sync overhead). Overlay one or more "
                            "analytic scaling laws to compare, and tune the serial fraction *s* "
                            "used by Amdahl/Gustafson."
                        )
                        msc, mslider = st.columns([3, 2])
                        with msc:
                            chosen = st.multiselect(
                                "Scaling laws to overlay",
                                list(SPEEDUP_MODELS.keys()),
                                default=["linear", "amdahl"],
                                format_func=lambda k: SPEEDUP_MODELS[k].name,
                                key="ddp_speedup_models",
                            )
                        with mslider:
                            serial_frac = st.slider(
                                "Serial fraction s (Amdahl / Gustafson)",
                                0.0, 0.5, 0.05, 0.01, key="ddp_serial_frac",
                            )

                        fig_su = go.Figure()
                        fig_su.add_trace(go.Scatter(
                            x=n_gpus_vals, y=speedup_vals,
                            name="Feasibility (compute/IO-aware)",
                            mode="lines+markers",
                            line=dict(color=COLORS[2], width=3),
                            marker=dict(size=10),
                        ))
                        _dash = {"linear": "dash", "amdahl": "dot", "gustafson": "dashdot"}
                        for i, key in enumerate(chosen):
                            fig_su.add_trace(go.Scatter(
                                x=n_gpus_vals,
                                y=speedup_curve(key, n_gpus_vals, serial_frac),
                                name=SPEEDUP_MODELS[key].name,
                                mode="lines+markers",
                                line=dict(color=COLORS[(i + 3) % len(COLORS)], width=2,
                                          dash=_dash.get(key, "dash")),
                            ))
                        fig_su.update_layout(
                            **_base_layout(340, "Speedup vs number of GPUs"),
                            xaxis_title="Number of GPUs",
                            yaxis_title="Speedup",
                        )
                        fig_su.update_xaxes(tickvals=n_gpus_vals)
                        _show(fig_su, "ddp_speedup")
                        if chosen:
                            st.caption("  ·  ".join(
                                f"**{SPEEDUP_MODELS[k].name}:** `{SPEEDUP_MODELS[k].formula}`"
                                for k in chosen
                            ))

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
                    _band_by_conf = {"high": 0.020, "medium": 0.035, "low": 0.050}
                    uncertainty = _band_by_conf.get(str(confidence).lower(), 0.035)
                    st.caption(
                        "**Empirical prior**, not a measurement: the expected Val F1 is anchored "
                        "to documented BigEarthNet-S2 runs of this model family and scaled to the "
                        "dataset size of this report. The band widens as confidence drops "
                        f"(here ±{uncertainty:.3f}, confidence **{confidence}**). For a measured "
                        "estimate use the convergence study below."
                    )

                    fig_pred = go.Figure()

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
                            "Blue line = empirical prior | "
                            "second line = real Val F1 of the selected run | "
                            "star = estimated best epoch"
                        )
                    else:
                        st.caption(
                            "Select a run in the sidebar to overlay the real results."
                        )

                # Prediction data as a downloadable table
                if curve_val and curve_epochs:
                    pred_curve_df = pd.DataFrame({
                        "epoch": curve_epochs,
                        "val_f1_pred": curve_val,
                        "train_f1_pred": curve_train if curve_train else [None] * len(curve_epochs),
                        "val_f1_upper": [v + uncertainty for v in curve_val],
                        "val_f1_lower": [v - uncertainty for v in curve_val],
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
        from src.gpu_specs import detect_all
        _gpus_avail = detect_all()
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
                feas_device = 0
                if len(_gpus_avail) > 1:
                    _dev_labels = {
                        f"cuda:{g.index} — {g.name} ({g.cuda_cores:,} CUDA cores)": g.index
                        for g in _gpus_avail
                    }
                    _sel = st.selectbox("GPU device", list(_dev_labels.keys()),
                                        help="Which GPU to run the benchmark on (multi-GPU host).")
                    feas_device = _dev_labels[_sel]
                elif len(_gpus_avail) == 1:
                    st.caption(f"GPU: cuda:0 — {_gpus_avail[0].name}")

                # Precision = Tensor-core switch (options gated by the GPU)
                from src.precision import available_precisions, label as _plabel
                _cc = _gpus_avail[0].compute_capability if _gpus_avail else None
                _precs = available_precisions(_cc, is_cuda=bool(_gpus_avail))
                feas_precision = st.selectbox(
                    "Precision (Tensor-core switch)", _precs,
                    format_func=_plabel,
                    help="fp32 = CUDA cores; tf32/amp/bf16 = Tensor cores (faster, less VRAM).",
                )
                feas_compare_prec = st.checkbox(
                    "Compare FP32 vs Tensor cores", value=False,
                    help="Run an extra FP32-vs-Tensor pass and report the speedup.",
                    disabled=len(_precs) <= 1,
                )
            submitted_feas = st.form_submit_button("Run")

        if submitted_feas:
            if not feas_batches:
                st.error("Select at least one batch size.")
            else:
                # Build an argv LIST and run WITHOUT shell=True so free-text fields
                # (e.g. the dataset path) can never be interpreted as shell syntax.
                argv = [
                    "uv", "run", "python", "scripts/check_feasibility.py",
                    "--model", feas_model,
                    "--batch-sizes", *[str(b) for b in feas_batches],
                    "--epochs", str(feas_epochs),
                    "--trace-modes", *(feas_traces if feas_traces else ["off"]),
                ]
                if feas_nfs != 1.0:
                    argv += ["--nfs-factor", str(feas_nfs)]
                if feas_config != "(none)":
                    argv += ["--config", f"configs/{feas_config}"]
                if feas_dataset_path.strip():
                    argv += ["--dataset-path", feas_dataset_path.strip()]
                if feas_no_disk:
                    argv.append("--no-disk-profile")
                if feas_study:
                    argv += ["--convergence-study", "--study-steps", str(feas_study_steps)]
                if feas_device:
                    argv += ["--device", str(feas_device)]
                if feas_precision and feas_precision != "fp32":
                    argv += ["--precision", feas_precision]
                if feas_compare_prec:
                    argv.append("--compare-precision")
                st.code(" ".join(shlex.quote(a) for a in argv), language="bash")
                out_ph = st.empty()
                with st.spinner("Running full analysis…"):
                    result = subprocess.run(argv, capture_output=True, text=True, cwd=str(ROOT))
                if result.returncode == 0:
                    st.success("Analysis complete.")
                    out_ph.code(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
                    _get_feasibility_csvs.clear()
                else:
                    st.error("Error during the analysis:")
                    out_ph.code(result.stderr[-2000:])



def _analytic_predictor() -> None:
    """Closed-form predictor: estimate time/speedup/memory/cost for ANY
    (strategy, model, GPU, n_gpus, dataset, batch, precision) from specs — no
    benchmark required. Powered by src/performance_model.py."""
    st.markdown("### Estimate a training before running it")
    st.caption(
        "Choose the parameters and get the full estimate — time, speedup, memory "
        "(and OOM), and cloud cost — from analytic formulas calibrated on real "
        "data, **without running anything**. Plan here first, then run the matching "
        "training and compare in Validate. Errors vs the real Kaggle 2×T4 runs: "
        "time +4%, DDP speedup <1%, AMP <2%."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Model", list(MODEL_TABLE.keys()),
                                  index=list(MODEL_TABLE.keys()).index("vit_base_patch16_224"))
        gpu_name = st.selectbox("GPU", list(GPU_TABLE.keys()),
                                index=list(GPU_TABLE.keys()).index("Tesla T4"))
    with c2:
        strategy = st.selectbox("Strategy", ["single", "ddp", "model_parallel", "heterogeneous"])
        n_gpus = st.number_input("Number of GPUs", 1, 8, 2 if strategy != "single" else 1)
    with c3:
        precision = st.selectbox("Precision", ["fp32", "amp", "tf32", "bf16"])
        disk_type = st.selectbox("Disk", ["ssd", "nvme", "hdd", "nfs"])

    c4, c5, c6 = st.columns(3)
    with c4:
        dataset_size = st.number_input("Train images / epoch", 100, 300000, 5000, step=500)
    with c5:
        batch = st.number_input("Global batch size", 1, 1024, 96, step=8)
    with c6:
        epochs = st.number_input("Epochs", 1, 200, 15)

    if strategy == "single":
        n_gpus = 1
    nfs = disk_type == "nfs"

    # ── Calibration bridge ─────────────────────────────────────────────────────
    # The whole point of the Measure tab is to close the loop: feed a measured
    # throughput (img/s, one GPU, fp32) back into the formula so the estimate is
    # grounded on THIS hardware. Default 0 = pure analytic prediction.
    with st.expander("Calibrate with a measured throughput (optional)"):
        st.caption(
            "Pure formulas assume a typical model-FLOPs utilization. If you have a real "
            "single-GPU throughput (from the **Measure** tab benchmark, fp32, img/s), enter "
            "it here to override the estimate for this exact hardware — the prediction is "
            "then flagged *calibrated*."
        )
        rc_meas = st.number_input("Measured throughput (img/s, 0 = none)", 0.0, 5000.0, 0.0,
                                  step=5.0, key="predict_rc_meas")
    rc_measured = rc_meas if rc_meas and rc_meas > 0 else None

    p = predict(strategy, model_name, gpu_name, n_gpus=int(n_gpus),
                dataset_size=int(dataset_size), batch=int(batch), precision=precision,
                epochs=int(epochs), disk_type=disk_type, nfs=nfs,
                rc_measured=rc_measured)
    if p is None:
        st.error("Unknown model or GPU spec.")
        return
    if p.calibrated:
        st.success("Calibrated with the measured throughput you entered.")

    st.markdown("#### Prediction")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train / epoch", _fmt_secs(p.time_per_epoch_train_s))
    m2.metric("Total train", _fmt_secs(p.time_total_train_s))
    if strategy != "single":
        m3.metric("Speedup", f"{p.speedup:.2f}×")
        m4.metric("Efficiency", f"{p.efficiency * 100:.0f}%")
    else:
        m3.metric("Throughput", f"{dataset_size / p.time_per_epoch_train_s:.0f} img/s")
        m4.metric("Bottleneck", p.bottleneck)

    m5, m6, m7 = st.columns(3)
    m5.metric("VRAM / GPU", f"{p.vram_per_gpu_gb:.1f} GB")
    m6.metric("Fits in memory", "yes" if p.fits_in_memory else "NO — OOM")
    m7.metric("Max batch that fits", p.recommended_batch)

    if not p.fits_in_memory:
        st.error(f"**Out of memory**: needs ~{p.vram_per_gpu_gb:.1f} GB but "
                 f"{gpu_spec(gpu_name).vram_gb:.0f} GB available. "
                 f"Largest batch that fits: **{p.recommended_batch}**.")
    for note in p.notes:
        st.info(note)

    # ── Estimated cloud cost for this exact configuration ──────────────────────
    from src.cloud_cost import estimate_costs
    total_h = p.time_total_train_s / 3600 * (n_gpus if strategy in ("ddp", "heterogeneous") else 1)
    cost_rows = estimate_costs(total_h, gpu_name)
    own = next((r for r in cost_rows if (gpu_name.split()[-1] in r["gpu"] or r["gpu"] in gpu_name)
                and r["usd_per_hour"] > 0), None)
    st.markdown("#### Estimated cost")
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        st.metric(f"On {gpu_name}",
                  f"${own['cost_usd']:.2f}" if own else "free / n/a",
                  help="GPU-hours × on-demand price for this exact run.")
        st.caption(f"≈ {total_h:.1f} GPU-hours")
    with cc2:
        cheap = [r for r in cost_rows if r["usd_per_hour"] > 0][:4]
        if cheap:
            cdf = pd.DataFrame([
                {"Provider": r["provider"], "GPU": r["gpu"],
                 "$/h": r["usd_per_hour"], "Cost ($)": r["cost_usd"]}
                for r in cheap
            ])
            st.dataframe(cdf, hide_index=True, use_container_width=True)
        st.caption("Cheapest paid options for an equivalent run (free tiers like "
                   "Kaggle/Colab omitted). Prices are a ballpark, editable in "
                   "`src/cloud_cost.py`.")

    # ── Expected quality (best Val F1) — the honest empirical prior ─────────────
    q = predict_quality(model_name, dataset_size=int(dataset_size), epochs=int(epochs))
    if q is not None:
        st.markdown("#### Expected quality")
        st.caption(
            "Best Val F1 you can expect, as an **empirical prior** anchored to our "
            "documented BigEarthNet-S2 runs and scaled to your dataset size by the "
            f"data-scaling law (more data → higher F1 with diminishing returns). "
            f"Confidence: **{q.confidence}**. This is a planning estimate, **not a "
            "measurement** — run the convergence study (Measure tab) for a measured one."
        )
        qc1, qc2, qc3 = st.columns(3)
        qc1.metric("Expected best Val F1", f"{q.expected_best_f1:.3f}",
                   delta=f"±{q.band:.3f}", delta_color="off")
        qc2.metric("Best epoch ≈", q.best_epoch)
        qc3.metric("Early stop ≈", q.early_stop_epoch, help="patience=10")

        ep = q.curve_epochs
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(
            x=ep + ep[::-1],
            y=[v + q.band for v in q.curve_val_f1] + [v - q.band for v in q.curve_val_f1[::-1]],
            fill="toself", fillcolor="rgba(34,114,180,0.10)",
            line=dict(color="rgba(255,255,255,0)"), name=f"Prior range (±{q.band:.3f})",
            hoverinfo="skip",
        ))
        fig_q.add_trace(go.Scatter(x=ep, y=q.curve_val_f1, name="Expected Val F1",
                                   mode="lines", line=dict(color=COLORS[0], width=3)))
        fig_q.add_trace(go.Scatter(x=ep, y=q.curve_train_f1, name="Expected Train F1",
                                   mode="lines", line=dict(color=COLORS[0], width=1.6, dash="dot"),
                                   opacity=0.6))
        if q.best_epoch <= max(ep):
            fig_q.add_trace(go.Scatter(
                x=[q.best_epoch], y=[q.curve_val_f1[q.best_epoch - 1]],
                name=f"Best epoch ({q.best_epoch})", mode="markers",
                marker=dict(color=COLORS[2], size=12, symbol="star"),
            ))
        fig_q.update_layout(**_base_layout(320, "Expected Val F1 curve (empirical prior)"),
                            xaxis_title="Epoch", yaxis_title="Val F1 (macro)")
        fig_q.update_yaxes(range=[0.0, 1.0])
        _show(fig_q, "predictor_quality")

    # Speedup curve across 1..8 GPUs (data-parallel scaling at a glance).
    if strategy in ("ddp", "heterogeneous"):
        st.markdown("#### Scaling 1 → 8 GPUs (predicted)")
        ns = [1, 2, 4, 8]
        sp = []
        for n in ns:
            pn = predict(strategy, model_name, gpu_name, n_gpus=n,
                         dataset_size=int(dataset_size), batch=int(batch),
                         precision=precision, epochs=1, disk_type=disk_type, nfs=nfs)
            sp.append(pn.speedup if pn else 0.0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ns, y=ns, name="Ideal (linear)",
                                 line=dict(color=COLORS[4], dash="dash")))
        fig.add_trace(go.Scatter(x=ns, y=sp, name="Predicted", mode="lines+markers",
                                 line=dict(color=COLORS[0], width=2), marker=dict(size=8)))
        fig.update_layout(**_base_layout(320, "Speedup vs number of GPUs"),
                          xaxis_title="GPUs", yaxis_title="Speedup ×")
        fig.update_xaxes(tickvals=ns)
        _show(fig, "predictor_scaling")
        st.caption("The curve flattens when the I/O total (fixed, shared disk) or "
                   "the gradient-sync term overtakes the per-GPU compute — exactly "
                   "the compute/IO/sync regimes of the analytic model.")


def _fmt_secs(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"
