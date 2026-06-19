"""Feasibility — validate."""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.feasibility_comparison import build_comparison
from src.web.feasibility_parser import (parse_ddp_scenarios, parse_feasibility_csv)
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.helpers import (_feas_label, _get_feasibility_csvs, _get_runs, _load_df)


def render_validate(ctx) -> object:
    selected_run = ctx.selected_run
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
    return subtab_prediction


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


