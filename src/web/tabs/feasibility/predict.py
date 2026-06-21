"""Feasibility — the analytic Predict tab (no GPU, just formulas).

Mirrors `tfg predict`: pick a config and get the full estimate — time and memory
with the formulas plugged in, expected quality, distributed scaling and cloud cost.
Powered by src/performance_model.py (no benchmark run)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.performance_model import (
    GPU_TABLE, MODEL_TABLE, predict, predict_quality, gpu_spec, model_spec,
    estimate_rc, estimate_rio, precision_factor,
    BYTES_PER_PARAM_GRAD_OPT, CUDA_OVERHEAD_GB, N_FULL_TRAIN, N_SUBSET_TRAIN,
)
from src.web.ui.charts import COLORS, _show, _base_layout, _dl_csv


def _fmt_secs(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def _analytic_predictor() -> None:
    """Closed-form predictor for ANY (strategy, model, GPU, n_gpus, dataset, batch,
    precision) — no benchmark. Shows the time/memory formulas with the values."""
    st.markdown("### Estimate a training before running it")
    st.caption(
        "Pick the parameters and get the full estimate — **with the formulas behind "
        "it** — from analytic models calibrated on real data, **without running "
        "anything**. Same engine as `tfg predict`. Errors vs the real Kaggle 2×T4 "
        "runs: time +4%, DDP speedup <1%, AMP <2%."
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
        _ds_choice = st.selectbox(
            "Dataset", [f"Subset ({N_SUBSET_TRAIN:,})", f"Full ({N_FULL_TRAIN:,})", "Custom"],
            help="Full BigEarthNet-S2 train split, the demo subset, or a custom count.")
        if _ds_choice.startswith("Full"):
            dataset_size = N_FULL_TRAIN
        elif _ds_choice.startswith("Subset"):
            dataset_size = N_SUBSET_TRAIN
        else:
            dataset_size = st.number_input("Custom train images / epoch", 100, 300000, 5000, step=500)
    with c5:
        batch = st.number_input("Global batch size", 1, 1024, 96, step=8)
    with c6:
        epochs = st.number_input("Epochs", 1, 200, 15)

    if strategy == "single":
        n_gpus = 1
    nfs = disk_type == "nfs"

    with st.expander("Calibrate with a measured throughput (optional)"):
        st.caption("Pure formulas assume a typical model-FLOPs utilization. Enter a real "
                   "single-GPU throughput (fp32, img/s, from a Measure-tab benchmark) to "
                   "ground the estimate on this exact hardware — it is then flagged *calibrated*.")
        rc_meas = st.number_input("Measured throughput (img/s, 0 = none)", 0.0, 5000.0, 0.0,
                                  step=5.0, key="predict_rc_meas")
    rc_measured = rc_meas if rc_meas and rc_meas > 0 else None

    p = predict(strategy, model_name, gpu_name, n_gpus=int(n_gpus),
                dataset_size=int(dataset_size), batch=int(batch), precision=precision,
                epochs=int(epochs), disk_type=disk_type, nfs=nfs, rc_measured=rc_measured)
    if p is None:
        st.error("Unknown model or GPU spec.")
        return
    ms, gs = model_spec(model_name), gpu_spec(gpu_name)

    # ── Headline strip ──────────────────────────────────────────────────────────
    if p.calibrated:
        st.success("Calibrated with the measured throughput you entered.")
    fit = "fits" if p.fits_in_memory else "**OOM**"
    sp = f" · {p.speedup:.2f}× ({p.efficiency*100:.0f}% eff)" if strategy != "single" else ""
    st.markdown(f"**{model_name.replace('_patch16_224','')} · {gpu_name} · {strategy} · "
                f"{precision}** → ~{_fmt_secs(p.time_per_epoch_train_s)}/epoch{sp} · "
                f"{p.vram_per_gpu_gb:.1f} GB/GPU ({fit}) · total "
                f"{_fmt_secs(p.time_total_train_s)} for {int(epochs)} ep · "
                f"bottleneck **{p.bottleneck}**")

    # ── Time / epoch formula ──────────────────────────────────────────────────────
    st.markdown("#### Time / epoch = max(compute, I/O) + sync")
    time_tbl = pd.DataFrame([
        {"Term": "Compute", "Value": f"{p.t_compute_s:.0f} s", "Formula": "N / (π · r_c · n_gpus)"},
        {"Term": "Data I/O", "Value": f"{p.t_io_s:.0f} s", "Formula": "N / r_io  (fixed, shared disk)"},
        {"Term": "Grad sync", "Value": f"{p.t_sync_s:.1f} s", "Formula": "(8·P / β) · n_batches"},
        {"Term": "Time / epoch", "Value": _fmt_secs(p.time_per_epoch_train_s),
         "Formula": f"max({p.t_compute_s:.0f}, {p.t_io_s:.0f}) + {p.t_sync_s:.1f} → {p.bottleneck}-bound"},
    ])
    st.dataframe(time_tbl, hide_index=True, use_container_width=True)

    # ── VRAM formula ──────────────────────────────────────────────────────────────
    if ms and gs:
        amp = precision in ("amp", "bf16")
        wo = BYTES_PER_PARAM_GRAD_OPT * ms.params_m * 1e6 / 1e9 + (2 * ms.params_m * 1e6 / 1e9 if amp else 0)
        act = ms.act_gb_per_img * p.batch_per_gpu * (0.6 if amp else 1.0)
        st.markdown("#### VRAM / GPU = weights+grad+optimizer + activations + overhead")
        mem_tbl = pd.DataFrame([
            {"Term": "Weights+grad+Adam", "Value": f"{wo:.2f} GB",
             "Formula": f"16 B × {ms.params_m:.0f}M" + (" + fp16 copy" if amp else "")},
            {"Term": "Activations", "Value": f"{act:.2f} GB",
             "Formula": f"{p.batch_per_gpu} × {ms.act_gb_per_img:.3f} GB/img" + (" × 0.6" if amp else "")},
            {"Term": "CUDA overhead", "Value": f"{CUDA_OVERHEAD_GB:.2f} GB", "Formula": "context + cudnn"},
            {"Term": "Total", "Value": f"{p.vram_per_gpu_gb:.1f} GB",
             "Formula": f"vs {gs.vram_gb:.0f} GB → {'fits' if p.fits_in_memory else 'OOM'} · "
                        f"max batch {p.recommended_batch}"},
        ])
        st.dataframe(mem_tbl, hide_index=True, use_container_width=True)
        if not p.fits_in_memory:
            st.error(f"Out of memory: needs ~{p.vram_per_gpu_gb:.1f} GB, GPU has "
                     f"{gs.vram_gb:.0f} GB. Largest batch that fits: **{p.recommended_batch}**.")

    # ── Expected quality (empirical prior) + curve ───────────────────────────────
    q = predict_quality(model_name, dataset_size=int(dataset_size), epochs=int(epochs))
    if q is not None:
        st.markdown("#### Expected quality (empirical prior — not a measurement)")
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
            line=dict(color="rgba(255,255,255,0)"), name=f"±{q.band:.3f}", hoverinfo="skip"))
        fig_q.add_trace(go.Scatter(x=ep, y=q.curve_val_f1, name="Expected Val F1",
                                   mode="lines", line=dict(color=COLORS[0], width=3)))
        fig_q.add_trace(go.Scatter(x=ep, y=q.curve_train_f1, name="Expected Train F1",
                                   mode="lines", line=dict(color=COLORS[0], width=1.6, dash="dot"),
                                   opacity=0.6))
        fig_q.update_layout(**_base_layout(300, "Expected Val F1 curve (empirical prior)"),
                            xaxis_title="Epoch", yaxis_title="Val F1 (macro)")
        fig_q.update_yaxes(range=[0.0, 1.0])
        _show(fig_q, "predictor_quality")

    # ── Distributed scaling 1→8 GPUs ─────────────────────────────────────────────
    if strategy in ("ddp", "heterogeneous"):
        st.markdown("#### Scaling 1 → 8 GPUs (predicted)")
        ns = [1, 2, 4, 8]
        sp_vals = []
        for n in ns:
            pn = predict(strategy, model_name, gpu_name, n_gpus=n, dataset_size=int(dataset_size),
                         batch=int(batch), precision=precision, epochs=1, disk_type=disk_type, nfs=nfs)
            sp_vals.append(pn.speedup if pn else 0.0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ns, y=ns, name="Ideal (linear)",
                                 line=dict(color="#94a3b8", dash="dash")))
        fig.add_trace(go.Scatter(x=ns, y=sp_vals, name="Predicted", mode="lines+markers",
                                 line=dict(color=COLORS[0], width=2), marker=dict(size=8)))
        fig.update_layout(**_base_layout(300, "Speedup vs number of GPUs"),
                          xaxis_title="GPUs", yaxis_title="Speedup ×")
        fig.update_xaxes(tickvals=ns)
        _show(fig, "predictor_scaling")
        st.caption("The curve flattens when the fixed I/O total or the gradient-sync term "
                   "overtakes the per-GPU compute.")

    # ── Cloud cost ───────────────────────────────────────────────────────────────
    from src.cloud_cost import estimate_costs
    total_h = p.time_total_train_s / 3600 * (n_gpus if strategy in ("ddp", "heterogeneous") else 1)
    rows = [r for r in estimate_costs(total_h, gpu_name) if r["usd_per_hour"] > 0][:6]
    if rows:
        st.markdown(f"#### Cloud cost · {total_h:.1f} GPU-hours")
        cdf = pd.DataFrame([{"Provider": r["provider"], "GPU": r["gpu"],
                             "$/h": r["usd_per_hour"], "Cost ($)": r["cost_usd"]} for r in rows])
        st.dataframe(cdf, hide_index=True, use_container_width=True)
        _dl_csv(cdf, "predicted_cost.csv", "Download cost table")

    # ── Assumptions ──────────────────────────────────────────────────────────────
    if ms and gs:
        rc = estimate_rc(ms, gs, precision)
        rio = estimate_rio(disk_type, nfs)
        st.caption(f"Assumptions: r_c ≈ {rc:.0f} img/s/GPU (incl. precision ×"
                   f"{precision_factor(gs, precision):.2f}) · r_io ≈ {rio:.0f} img/s "
                   f"({disk_type}{'+NFS' if nfs else ''}) · {ms.params_m:.0f}M params · MFU 0.17.")
    for note in p.notes:
        st.info(note)
