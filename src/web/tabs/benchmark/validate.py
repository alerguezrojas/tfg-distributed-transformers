"""Benchmark — validate (predicted vs actual, Compare-style)."""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.benchmark_comparison import build_comparison
from src.web.benchmark_parser import (parse_ddp_scenarios, parse_benchmark_csv)
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.helpers import (_get_benchmark_csvs, _get_runs, _load_df, _run_config,
                                _safe_max)


def _short(lbl: str) -> str:
    return re.sub(r"^\d{2}/\d{2}/\d{4}\s+", "", lbl)


def _run_batch(r) -> int | None:
    """Per-GPU batch the run used (from its config line) — to match the benchmark
    row of the SAME batch (the benchmark is single-GPU, at that per-GPU batch)."""
    m = re.search(r"\d+", str(_run_config(str(r.log_path)).get("batch", "")))
    return int(m.group()) if m else None


def _run_ngpus(r) -> int:
    """Number of GPUs a distributed run used: global ÷ per-GPU batch if recorded,
    else 2 (the default for the project's DDP runs)."""
    txt = str(_run_config(str(r.log_path)).get("batch", ""))
    per = re.search(r"\d+", txt)
    glob = re.search(r"global\s*=?\s*(\d+)", txt)
    if per and glob and int(per.group()) > 0:
        return max(1, round(int(glob.group(1)) / int(per.group())))
    return 2


def _predicted_speedup(meta, n_gpus) -> float | None:
    """Predicted speedup at n_gpus from the benchmark report's DDP scenarios."""
    scen = parse_ddp_scenarios(meta)
    if not scen.empty and {"n_gpus", "speedup"}.issubset(scen.columns):
        row = scen[scen["n_gpus"] == n_gpus]
        if not row.empty:
            return float(row.iloc[0]["speedup"])
    return None


def _analytic_speedup(r, meta) -> float | None:
    """Speedup vs single-GPU from the analytic engine, for the strategies the benchmark
    does NOT model: model-parallel (naive pipeline ≈1×) and heterogeneous. The empirical
    benchmark only extrapolates data-parallel (DDP) scaling, so without this an MP run
    would have no estimate here. Uses the GPU recorded in the run's benchmark report."""
    gpu = (meta or {}).get("hardware_name")
    bs = _run_batch(r)
    if not gpu or not bs:
        return None
    try:
        from src.performance_model import predict
        p = predict(r.mode, r.model, gpu, _run_ngpus(r), batch=bs,
                    precision=(r.precision or "fp32"))
        return float(p.speedup) if p and p.speedup else None
    except Exception:
        return None


def _real_min_per_epoch(r) -> float | None:
    df = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
    if "epoch_time" in df.columns and df["epoch_time"].notna().any():
        return float(df["epoch_time"].mean()) / 60.0
    return None


def render_validate(ctx) -> object:
    """Predicted vs actual, Compare-style. Each single-GPU run is matched to the
    benchmark report of its model AND the batch size it actually used, so the
    estimate is for the same configuration the run ran (this is the fix for the
    'estimate very different from real' issue — the old code used the max-throughput
    batch, not the run's). Shows a table, a scorecard, a calibration scatter, the
    formula behind each estimate, and predicted-vs-real speedup."""
    st.markdown("### Predicted vs actual")
    st.caption("Pick runs and compare what the benchmark **estimated** with what they "
               "actually did. Each single-GPU run is matched to the benchmark report "
               "of its model **and its batch size**, so the estimate is for the same "
               "configuration the run used.")

    feas_csvs = _get_benchmark_csvs()
    if not feas_csvs:
        st.info("No benchmark reports yet. Generate one from the terminal "
                "(`paravit benchmark`).")
        st.divider()
        return st.container()

    parsed = []
    for p in feas_csvs:
        m, df = parse_benchmark_csv(p)
        env = p.parent.parent.name if p.parent.parent else "?"
        parsed.append((env, m.get("model_name", "?"), m, df))

    def _feas_for(env, model):
        same = [(m, df) for e, mo, m, df in parsed if mo == model and e == env]
        if same:
            return same[0]
        any_m = [(m, df) for e, mo, m, df in parsed if mo == model]
        return any_m[0] if any_m else (None, None)

    runs = _get_runs()
    labelled = {r.label: r for r in runs}
    feas_models = {mo for _, mo, _, _ in parsed}
    # Default prefers single-GPU runs (they get a full per-metric comparison) plus a
    # couple of distributed ones, so the table and the formula picker are populated.
    _sing = [r.label for r in runs if r.mode == "single" and r.model in feas_models]
    _oth = [r.label for r in runs if r.mode != "single" and r.model in feas_models]
    default = (_sing[:5] + _oth[:2])[:8] or list(labelled)[:3]
    sel = st.multiselect("Runs to compare against their estimate (max 8)",
                         list(labelled.keys()), default=default, max_selections=8)
    if not sel:
        st.info("Select at least one run.")
        st.divider()
        return st.container()

    # ── Per-run comparison (single-GPU runs matched by batch via build_comparison) ─
    # Apples-to-apples requires the SAME precision: the benchmark report is fp32,
    # so an AMP run (Tensor cores) is ~3× faster than its fp32 estimate — that gap is
    # the precision, not a prediction error. We flag those so the table is honest.
    rows = []
    cmp_by_run: dict = {}
    for lbl in sel:
        r = labelled[lbl]
        run_df = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
        real_min = _real_min_per_epoch(r)
        real_f1 = _safe_max(run_df["val_f1"]) if "val_f1" in run_df.columns else float("nan")
        m, fdf = _feas_for(r.env, r.model)
        bs = _run_batch(r)
        run_prec = (r.precision or "fp32")
        # Measured fp32→Tensor-core speedup from the report's --compare-precision block.
        _pc = (m.get("precision_cmp") or {}) if m else {}
        try:
            prec_sp = float(_pc.get("speedup")) if _pc.get("speedup") else None
        except (TypeError, ValueError):
            prec_sp = None
        tensor_core = run_prec in ("amp", "fp16", "tf32", "bf16")
        # We can account for precision iff fp32 (nothing to do) or the report measured
        # the Tensor-core speedup. Otherwise the AMP run is not comparable to fp32.
        precision_ok = (not tensor_core) or (prec_sp is not None)
        est_min = None
        note_bits = []
        if m and fdf is not None and not fdf.empty and bs and r.mode in (
                "single", "ddp", "model_parallel", "heterogeneous"):
            nfs = float(m.get("nfs_factor", 1.0) or 1.0)
            cmp = build_comparison(meta=m, feas_df=fdf, actual_df=run_df,
                                   batch_size=bs, trace_mode="simple", nfs_factor=nfs)
            if cmp:
                tt = next((x for x in cmp.rows if x.metric == "Total time / epoch"), None)
                single_est = tt.estimated if (tt and tt.estimated is not None) else None
                if r.mode == "single":
                    cmp_by_run[lbl] = cmp        # formula table is the single-GPU one
                # Apply the measured Tensor-core speedup so an AMP run is compared to an
                # AMP estimate (fp32 estimate ÷ measured speedup), not the fp32 one.
                if single_est is not None and tensor_core and prec_sp:
                    single_est = single_est / prec_sp
                    note_bits.append(f"fp32 ÷ {prec_sp:.1f}× ({run_prec})")
                if single_est is not None:
                    if r.mode == "single":
                        est_min = single_est
                    elif r.mode == "ddp":        # ÷ predicted N-GPU speedup (benchmark)
                        ng = _run_ngpus(r)
                        sp = _predicted_speedup(m, ng)
                        if sp:
                            est_min = single_est / sp
                            note_bits.append(f"÷ {sp:.2f}× ({ng} GPU)")
                    else:                        # model-parallel / heterogeneous:
                        # the benchmark has no scenario for these, so the speedup comes
                        # from the analytic engine (model-parallel ≈1×).
                        sp = _analytic_speedup(r, m)
                        if sp:
                            est_min = single_est / sp
                            note_bits.append(f"÷ {sp:.2f}× (analytic {r.mode})")
        # Pure-analytic fallback for model-parallel / heterogeneous runs whose batch was
        # NOT benchmarked (e.g. vit_large MP at batch 24 — its report only covers 32/48/64),
        # so there is no single-GPU estimate to anchor. The analytic engine predicts the
        # run directly (train time/epoch); flagged as train-only since it omits eval.
        if est_min is None and r.mode in ("model_parallel", "heterogeneous") and m and bs:
            gpu = m.get("hardware_name")
            if gpu:
                try:
                    from src.performance_model import predict
                    p = predict(r.mode, r.model, gpu, _run_ngpus(r), batch=bs,
                                precision=run_prec)
                    if p and p.time_per_epoch_train_s:
                        est_min = p.time_per_epoch_train_s / 60.0
                        note_bits.append(f"analytic ({r.mode}, batch {bs} not "
                                         f"benchmarked, train-only)")
                except Exception:
                    pass
        pred_f1 = None
        if m:
            try:
                pred_f1 = float(m.get("prediction", {}).get("predicted_best_f1") or 0) or None
            except (TypeError, ValueError):
                pred_f1 = None
        fair = precision_ok and est_min is not None
        if est_min is None:
            note = ""
        elif tensor_core and not prec_sp:
            note = f"≠ precision ({run_prec}, not benchmarked)"
        else:
            note = " · ".join(note_bits)
        err = ((real_min - est_min) / est_min * 100) if (real_min and est_min) else None
        rows.append({
            "Run": _short(lbl),          # the run label already shows the model
            "Strategy": r.mode,
            "Batch": bs,
            "Precision": run_prec,
            "Est min/ep": round(est_min, 2) if est_min else None,
            "Real min/ep": round(real_min, 2) if real_min else None,
            "Time err %": round(err) if err is not None else None,
            "Pred F1": round(pred_f1, 3) if pred_f1 else None,
            "Real F1": round(float(real_f1), 3) if not pd.isna(real_f1) else None,
            "Note": note,
            "_fair": fair,
        })
    tdf = pd.DataFrame(rows).set_index("Run")
    _shown = tdf.drop(columns=["_fair"])
    st.dataframe(_shown, use_container_width=True,
                 column_config={"Note": st.column_config.TextColumn("Note", width="large")})
    _dl_csv(_shown.reset_index(), "predicted_vs_real.csv", "Download comparison")
    st.caption("Single-GPU runs are matched by batch size; **DDP** runs use single-GPU "
               "estimate ÷ predicted speedup; **model-parallel / heterogeneous** runs use "
               "the *analytic* engine's speedup (≈1× for naive pipeline) since the empirical "
               "benchmark only models data-parallel scaling; **AMP/TF32** runs are corrected "
               "by the report's *measured* Tensor-core speedup (see the *Note* column), so "
               "they are comparable. A precision the report never benchmarked is flagged "
               "*≠ precision*. The remaining gap is the genuine error — the synthetic "
               "benchmark omits disk I/O, so small models on NFS run a bit slower. For "
               "comparable compute-bound runs the estimate lands within a few %.")

    # ── Accuracy scorecard — over the apples-to-apples (same-precision) runs ─────
    fair_df = tdf[tdf["_fair"]]
    _terr = fair_df["Time err %"].dropna().abs()
    _f1p = tdf.dropna(subset=["Pred F1", "Real F1"])
    _f1err = (_f1p["Pred F1"] - _f1p["Real F1"]).abs() if not _f1p.empty else pd.Series(dtype=float)
    sc1, sc2 = st.columns(2)
    sc1.metric("Mean time error (comparable runs)",
               f"±{_terr.mean():.0f}%" if not _terr.empty else "—",
               help="Mean |error| of the batch-matched time/epoch over runs the report "
                    "can account for (fp32, or AMP/TF32 with a measured Tensor-core speedup).")
    sc2.metric("Mean F1 error", f"±{_f1err.mean():.3f}" if not _f1err.empty else "—",
               help="Mean |predicted − real| best Val F1 over the selected runs.")

    # ── Chart 1: estimated vs real time/epoch per run (grouped bars) ────────────
    bars = tdf.dropna(subset=["Est min/ep", "Real min/ep"])
    if not bars.empty:
        figb = go.Figure()
        figb.add_trace(go.Bar(name="Estimated", x=list(bars.index), y=list(bars["Est min/ep"]),
                              marker_color="#94a3b8",
                              text=[f"{v:.2f}" for v in bars["Est min/ep"]], textposition="outside"))
        figb.add_trace(go.Bar(name="Real", x=list(bars.index), y=list(bars["Real min/ep"]),
                              marker_color=COLORS[0],
                              text=[f"{v:.2f}" for v in bars["Real min/ep"]], textposition="outside"))
        figb.update_layout(**_base_layout(340, "Time per epoch — estimated vs real (single-GPU)"),
                           barmode="group", yaxis_title="Minutes", xaxis_title="")
        _show(figb, "validate_time_bars")

    # ── Chart 2: calibration scatter (predicted vs real, with the diagonal) ─────
    metric = st.radio("Calibration plot", ["Time/epoch (min)", "Val F1"],
                      horizontal=True, key="cal_metric")
    _xc, _yc, _unit = (("Est min/ep", "Real min/ep", "min/epoch")
                       if metric.startswith("Time") else ("Pred F1", "Real F1", "Val F1"))
    cal = tdf.dropna(subset=[_xc, _yc])
    if not cal.empty:
        _hi = (float(max(cal[_xc].max(), cal[_yc].max())) * 1.1) or 1.0
        _fair_pts = cal[cal["_fair"]]
        _unfair_pts = cal[~cal["_fair"]]
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=[0, _hi], y=[0, _hi], mode="lines",
                                  line=dict(color="#94a3b8", dash="dash"),
                                  name="perfect estimate", hoverinfo="skip"))
        for _pts, _name, _col in ((_fair_pts, "comparable", COLORS[0]),
                                  (_unfair_pts, "≠ precision (not benchmarked)", "#C57B27")):
            if not _pts.empty:
                figc.add_trace(go.Scatter(
                    x=_pts[_xc], y=_pts[_yc], mode="markers+text",
                    text=list(_pts.index), textposition="top center", textfont=dict(size=9),
                    marker=dict(size=12, color=_col, line=dict(width=1, color="white")),
                    name=_name,
                    hovertemplate="%{text}<br>estimated %{x:.2f}<br>real %{y:.2f}<extra></extra>"))
        figc.update_layout(**_base_layout(360, f"Estimated vs real — {_unit}"),
                           xaxis_title=f"Estimated ({_unit})", yaxis_title=f"Real ({_unit})")
        _show(figc, "validate_calibration")
        st.caption("On the dashed diagonal = perfect estimate. **Blue** = comparable "
                   "(fp32, or AMP corrected by the measured Tensor-core speedup); "
                   "**amber** = a precision the report did not benchmark.")
    else:
        st.caption(f"Not enough runs with an estimated and real {_unit} for the plot.")

    # ── Formula behind each estimate (the recovered detail table) ───────────────
    if cmp_by_run:
        st.markdown("#### Formula behind each estimate")
        st.caption(f"Per-metric formula + estimated-vs-real, for one **single-GPU** run. "
                   f"It lists the {len(cmp_by_run)} selected single-GPU run(s) whose **batch "
                   f"size was benchmarked** in their model's benchmark report (e.g. the "
                   f"vit_base reports cover batch 48/64/96; a run at a batch the report "
                   f"never measured can't be broken down here). Select more such runs above "
                   f"to see them.")
        pick = st.selectbox("Run (single-GPU)", list(cmp_by_run.keys()),
                            format_func=_short, key="formula_run")
        ftab = cmp_by_run[pick].to_dataframe()
        st.dataframe(ftab, hide_index=True, use_container_width=True)
        _dl_csv(ftab, "benchmark_formulas.csv", "Download formulas")

    # ── Speedup validation when a single + DDP pair of the same model is picked ──
    groups: dict = defaultdict(dict)
    for lbl in sel:
        r = labelled[lbl]
        if r.mode in ("single", "ddp"):
            groups[(r.env, r.model)][r.mode] = r
    sp_lines = []
    for (env, model), d in groups.items():
        if "single" in d and "ddp" in d:
            s, dd = _real_min_per_epoch(d["single"]), _real_min_per_epoch(d["ddp"])
            if s and dd:
                real_sp = s / dd
                m = next((mm for e, mo, mm, _ in parsed if e == env and mo == model),
                         next((mm for e, mo, mm, _ in parsed if mo == model), None))
                pred = None
                if m is not None:
                    scen = parse_ddp_scenarios(m)
                    if not scen.empty and {"n_gpus", "speedup"}.issubset(scen.columns):
                        r2 = scen[scen["n_gpus"] == 2]
                        if not r2.empty:
                            pred = float(r2.iloc[0]["speedup"])
                pre = f"predicted **{pred:.2f}×** · " if pred else ""
                sp_lines.append(f"- **{model.replace('_patch16_224','')}** ({env}): "
                                f"{pre}real **{real_sp:.2f}×** (2 GPUs)")
    if sp_lines:
        st.markdown("**Speedup at 2 GPUs — predicted vs real**")
        st.markdown("\n".join(sp_lines))

    st.divider()
    return st.container()


def render_f1_prediction(meta, selected_run, benchmark_csvs) -> None:
    if not benchmark_csvs:
        st.info("Run the benchmark analysis first.")
    else:
        st.markdown("## Empirical performance prediction")
        pred = meta.get("prediction", {})
        curve_val = meta.get("curve_val_f1", [])
        curve_train = meta.get("curve_train_f1", [])
        curve_epochs = meta.get("curve_epochs", [])

        if not pred:
            st.info(
                "No prediction data in this report. "
                "Regenerate with the current version of benchmark.py."
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
