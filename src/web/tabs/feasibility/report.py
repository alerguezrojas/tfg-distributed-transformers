"""Feasibility — report (elegant viewer of a benchmark generated in the terminal).

Summary strip + the key visuals up front (throughput, VRAM, precision, time, cost,
distributed scaling); the verbose system/memory tables live in one expander."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.helpers import (_throughput_col)


def render_report(meta, bdf_feas, feasibility_csvs) -> object:
    if not feasibility_csvs:
        st.info("No feasibility CSVs found. Generate one from the terminal "
                "(`tfg feasibility`).")
        return st.container()

    viable = bdf_feas[bdf_feas["oom"] == "no"].copy() if not bdf_feas.empty else bdf_feas
    tp_col = _throughput_col(viable) if not viable.empty else None

    # ── Summary strip (headline, always visible) ────────────────────────────────
    best_tp = float(viable[tp_col].max()) if tp_col and not viable.empty else None
    max_batch = (int(viable["batch_size"].max())
                 if "batch_size" in viable.columns and not viable.empty else None)
    per_ep = next((c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                   if not bdf_feas.empty and c in bdf_feas.columns), None)
    best_min = float(viable[per_ep].min()) if per_ep and not viable.empty else None
    try:
        io_ratio = float(meta.get("dataset", {}).get("io_bottleneck_ratio", 0) or 0)
    except (TypeError, ValueError, AttributeError):
        io_ratio = 0.0

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("GPU", meta.get("hardware_name", "—"))
    s2.metric("Best throughput", f"{best_tp:.0f} img/s" if best_tp else "—")
    s3.metric("Fastest epoch", f"{best_min:.1f} min" if best_min else "—")
    s4.metric("Max batch (fits)", max_batch if max_batch else "—")
    s5.metric("Bottleneck",
              "I/O-bound" if io_ratio > 1.2 else ("Compute-bound" if io_ratio else "—"))

    # ── Key visuals: throughput + peak VRAM side by side ────────────────────────
    if not viable.empty and tp_col:
        g1, g2 = st.columns(2)
        with g1:
            fig_tp = go.Figure()
            for mode_feas in viable["trace_mode"].unique():
                sub = viable[viable["trace_mode"] == mode_feas]
                fig_tp.add_trace(go.Bar(x=sub["batch_size"].astype(str), y=sub[tp_col],
                                        name=f"trace={mode_feas}"))
            fig_tp.update_layout(**_base_layout(300, "Throughput by batch size"),
                                 barmode="group", xaxis_title="Batch size", yaxis_title="img/s")
            _show(fig_tp, "throughput")
        with g2:
            if "peak_vram_gb" in viable.columns and viable["peak_vram_gb"].notna().any():
                fig_v = go.Figure()
                for mode_feas in viable["trace_mode"].unique():
                    sub = viable[viable["trace_mode"] == mode_feas]
                    fig_v.add_trace(go.Bar(x=sub["batch_size"].astype(str),
                                           y=sub["peak_vram_gb"], name=f"trace={mode_feas}"))
                if meta.get("free_vram_gb"):
                    fig_v.add_hline(y=float(meta["free_vram_gb"]), line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Free VRAM {meta['free_vram_gb']} GB",
                                    annotation_position="top left")
                fig_v.update_layout(**_base_layout(300, "Peak VRAM by batch size"),
                                    barmode="group", xaxis_title="Batch size", yaxis_title="GB")
                _show(fig_v, "peak_vram")
            else:
                st.caption("No peak-VRAM measurement in this report.")

    # ── Precision (Tensor cores) comparison, if measured ────────────────────────
    pc = meta.get("precision_cmp")
    if pc and pc.get("fp32_imgs_s"):
        tc_name = str(pc.get("tc_precision", "amp")).upper()
        fig_pc = go.Figure(go.Bar(
            x=[f"FP32 (CUDA cores)", f"{tc_name} (Tensor cores)"],
            y=[pc["fp32_imgs_s"], pc["tc_imgs_s"]],
            marker_color=[COLORS[5], COLORS[2]],
            text=[f"{pc['fp32_imgs_s']:.0f}", f"{pc['tc_imgs_s']:.0f}"], textposition="outside",
        ))
        fig_pc.update_layout(**_base_layout(240, f"Precision: CUDA vs Tensor cores "
                                            f"({pc['speedup']:.2f}× faster)"),
                             yaxis_title="img/s")
        _show(fig_pc, "precision_cmp")

    # ── Time estimate: total for N epochs at the fastest batch (no redundant chart) ─
    n_ep = 30
    if per_ep and not viable.empty:
        best = float(viable[per_ep].min())
        n_ep = st.number_input("Epochs for the total estimate", min_value=1, value=30,
                               key="report_epochs")
        st.metric(f"Estimated total ({n_ep} epochs, fastest batch)",
                  f"{best * n_ep / 60:.1f} h", help=f"{best:.2f} min/epoch × {n_ep} epochs")

    # ── Distributed scaling — filled by render_ddp_analysis (visible) ───────────
    st.markdown("#### Distributed scaling (predicted)")
    subtab_ddp_opt = st.container()

    # ── Verbose detail in ONE expander (system, memory, raw tables, cloud cost) ─
    with st.expander("System, memory, raw tables & cloud cost"):
        gpu = meta.get("gpu", {})
        st.markdown(f"**{meta.get('model_name','—')}** · {meta.get('total_params_M','—')} M "
                    f"params · {meta.get('hardware_name','—')} · "
                    f"{meta.get('total_vram_gb','—')} GB VRAM")
        if gpu and gpu.get("cuda_cores"):
            st.caption(f"{gpu.get('architecture','—')} · CC {gpu.get('compute_capability','—')} · "
                       f"{gpu.get('sm_count','—')} SMs · {int(gpu['cuda_cores']):,} CUDA cores · "
                       f"{int(gpu.get('tensor_cores',0)):,} Tensor cores")
        cpu = meta.get("cpu", {})
        if cpu:
            st.caption(f"CPU: {cpu.get('logical_cores','—')} logical / "
                       f"{cpu.get('physical_cores','—')} physical cores · "
                       f"{cpu.get('ram_total_gb','—')} GB RAM")
        disk = meta.get("disk", {})
        if disk:
            st.caption(f"Disk: {disk.get('type','—')} · NFS {disk.get('is_nfs','no')} · "
                       f"{disk.get('read_mb_per_s','—')} MB/s")
        mem_keys = ["weight_mb", "gradient_mb", "optimizer_mb", "activation_mb_per_image"]
        if any(k in meta for k in mem_keys):
            st.caption(f"Memory: weights {meta.get('weight_mb','—')} MB · grads "
                       f"{meta.get('gradient_mb','—')} MB · AdamW {meta.get('optimizer_mb','—')} MB · "
                       f"activations {meta.get('activation_mb_per_image','—')} MB/img")
        if not bdf_feas.empty:
            st.dataframe(bdf_feas, use_container_width=True, height=240)
            _dl_csv(bdf_feas, "feasibility_benchmark.csv", "Download full benchmark")

        # Cloud cost — uses the same epoch count as the estimate above
        if per_ep and not viable.empty:
            from src.cloud_cost import estimate_costs
            total_h = float(viable[per_ep].min()) * n_ep / 60.0
            rows = estimate_costs(total_h, meta.get("hardware_name", ""))
            paid = [r for r in rows if r["cost_usd"] > 0][:6]
            if paid:
                st.markdown(f"**Cloud training cost ({n_ep} epochs)**")
                fig_c = go.Figure(go.Bar(
                    x=[f"{r['provider']} · {r['gpu']}" for r in paid],
                    y=[r["cost_usd"] for r in paid], marker_color=COLORS[0],
                    text=[f"${r['cost_usd']:.0f}" for r in paid], textposition="outside"))
                fig_c.update_layout(**_base_layout(260, "Estimated cost by provider (paid)"),
                                    xaxis_tickangle=30, yaxis_title="USD")
                _show(fig_c, "cloud_cost")

    return subtab_ddp_opt
