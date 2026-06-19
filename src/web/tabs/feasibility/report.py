"""Feasibility — report."""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.helpers import (_throughput_col)


def render_report(meta, bdf_feas, feasibility_csvs) -> object:
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

    return subtab_ddp_opt

