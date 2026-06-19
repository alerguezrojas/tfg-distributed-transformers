"""Feasibility — the analytic Predict tab (no GPU, just formulas).

Closed-form predictor: time, memory/OOM, cloud cost and expected quality for any
(strategy, model, GPU, n_gpus, dataset, batch, precision). Powered by
src/performance_model.py. Kept in its own module so the Feasibility page stays
navigable."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.performance_model import (
    GPU_TABLE, MODEL_TABLE, predict, predict_quality, gpu_spec,
    N_FULL_TRAIN, N_SUBSET_TRAIN,
)
from src.web.ui.charts import COLORS, _show, _base_layout


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
        _ds_choice = st.selectbox(
            "Dataset",
            [f"Subset ({N_SUBSET_TRAIN:,})", f"Full ({N_FULL_TRAIN:,})", "Custom"],
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
