"""Feasibility — validate (predicted vs actual, Compare-style)."""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.feasibility_comparison import build_comparison
from src.web.feasibility_parser import (parse_ddp_scenarios, parse_feasibility_csv)
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.helpers import (_get_feasibility_csvs, _get_runs, _load_df, _run_config,
                                _safe_max)


def _short(lbl: str) -> str:
    return re.sub(r"^\d{2}/\d{2}/\d{4}\s+", "", lbl)


def _run_batch(r) -> int | None:
    """Batch size the run actually used (from its config line) — to match the
    feasibility row of the SAME batch (apples-to-apples)."""
    m = re.search(r"\d+", str(_run_config(str(r.log_path)).get("batch", "")))
    return int(m.group()) if m else None


def _real_min_per_epoch(r) -> float | None:
    df = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
    if "epoch_time" in df.columns and df["epoch_time"].notna().any():
        return float(df["epoch_time"].mean()) / 60.0
    return None


def render_validate(ctx) -> object:
    """Predicted vs actual, Compare-style. Each single-GPU run is matched to the
    feasibility report of its model AND the batch size it actually used, so the
    estimate is for the same configuration the run ran (this is the fix for the
    'estimate very different from real' issue — the old code used the max-throughput
    batch, not the run's). Shows a table, a scorecard, a calibration scatter, the
    formula behind each estimate, and predicted-vs-real speedup."""
    st.markdown("### Predicted vs actual")
    st.caption("Pick runs and compare what the feasibility **estimated** with what they "
               "actually did. Each single-GPU run is matched to the feasibility report "
               "of its model **and its batch size**, so the estimate is for the same "
               "configuration the run used.")

    feas_csvs = _get_feasibility_csvs()
    if not feas_csvs:
        st.info("No feasibility reports yet. Generate one from the terminal "
                "(`tfg feasibility`).")
        st.divider()
        return st.container()

    parsed = []
    for p in feas_csvs:
        m, df = parse_feasibility_csv(p)
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
    default = [r.label for r in runs if r.model in feas_models][:6] or list(labelled)[:3]
    sel = st.multiselect("Runs to compare against their estimate (max 8)",
                         list(labelled.keys()), default=default, max_selections=8)
    if not sel:
        st.info("Select at least one run.")
        st.divider()
        return st.container()

    # ── Per-run comparison (single-GPU runs matched by batch via build_comparison) ─
    # Apples-to-apples requires the SAME precision: the feasibility report is fp32,
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
        rep_prec = (m.get("precision") or "fp32") if m else "fp32"
        est_min = None
        if m and fdf is not None and not fdf.empty and bs and r.mode == "single":
            nfs = float(m.get("nfs_factor", 1.0) or 1.0)
            cmp = build_comparison(meta=m, feas_df=fdf, actual_df=run_df,
                                   batch_size=bs, trace_mode="simple", nfs_factor=nfs)
            if cmp:
                cmp_by_run[lbl] = cmp
                tt = next((x for x in cmp.rows if x.metric == "Total time / epoch"), None)
                est_min = tt.estimated if (tt and tt.estimated is not None) else None
        pred_f1 = None
        if m:
            try:
                pred_f1 = float(m.get("prediction", {}).get("predicted_best_f1") or 0) or None
            except (TypeError, ValueError):
                pred_f1 = None
        fair = (run_prec == rep_prec)          # comparable only at the same precision
        note = "" if (est_min is None or fair) else f"≠ precision ({run_prec} vs {rep_prec})"
        err = ((real_min - est_min) / est_min * 100) if (real_min and est_min) else None
        rows.append({
            "Run": _short(lbl),
            "Model": (r.model or "—").replace("_patch16_224", ""),
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
    st.dataframe(tdf.drop(columns=["_fair"]), use_container_width=True)
    _dl_csv(tdf.drop(columns=["_fair"]).reset_index(), "predicted_vs_real.csv", "Download comparison")
    st.caption("Time estimate only for **single-GPU** runs, matched by batch size. "
               "The feasibility benchmark is **fp32 and synthetic (no disk I/O)**, so two "
               "gaps are expected and flagged: **AMP** runs use Tensor cores (~3× faster "
               "than the fp32 estimate — *≠ precision*), and **small models on NFS** read "
               "more from disk than the benchmark models (real a bit slower). For "
               "same-precision compute-bound runs the estimate is within a few %.")

    # ── Accuracy scorecard — over the apples-to-apples (same-precision) runs ─────
    fair_df = tdf[tdf["_fair"]]
    _terr = fair_df["Time err %"].dropna().abs()
    _f1p = tdf.dropna(subset=["Pred F1", "Real F1"])
    _f1err = (_f1p["Pred F1"] - _f1p["Real F1"]).abs() if not _f1p.empty else pd.Series(dtype=float)
    sc1, sc2 = st.columns(2)
    sc1.metric("Mean time error (same precision)",
               f"±{_terr.mean():.0f}%" if not _terr.empty else "—",
               help="Mean |error| of the batch-matched time/epoch over runs whose "
                    "precision matches the fp32 benchmark (apples-to-apples).")
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
        for _pts, _name, _col in ((_fair_pts, "same precision", COLORS[0]),
                                  (_unfair_pts, "≠ precision", "#C57B27")):
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
        st.caption("On the dashed diagonal = perfect estimate. **Blue** = same precision "
                   "(apples-to-apples); **amber** = AMP run vs fp32 estimate (the offset "
                   "is the Tensor-core speedup, not an error).")
    else:
        st.caption(f"Not enough runs with an estimated and real {_unit} for the plot.")

    # ── Formula behind each estimate (the recovered detail table) ───────────────
    if cmp_by_run:
        st.markdown("#### Formula behind each estimate")
        st.caption("The exact formula and the estimated-vs-real value per metric, for "
                   "one single-GPU run (matched by its batch size).")
        pick = st.selectbox("Run", list(cmp_by_run.keys()),
                            format_func=_short, key="formula_run")
        ftab = cmp_by_run[pick].to_dataframe()
        st.dataframe(ftab, hide_index=True, use_container_width=True)
        _dl_csv(ftab, "feasibility_formulas.csv", "Download formulas")

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


def render_f1_prediction(meta, selected_run, feasibility_csvs) -> None:
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
