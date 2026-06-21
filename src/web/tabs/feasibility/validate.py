"""Feasibility — validate."""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.feasibility_parser import (parse_ddp_scenarios, parse_feasibility_csv)
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _show)
from src.web.ui.helpers import (_get_feasibility_csvs, _get_runs, _load_df, _safe_max,
                                _throughput_col)


def _short(lbl: str) -> str:
    return re.sub(r"^\d{2}/\d{2}/\d{4}\s+", "", lbl)


def _est_min_per_epoch(feas_df) -> float | None:
    """Estimated minutes/epoch from a feasibility report (best viable throughput)."""
    if feas_df is None or feas_df.empty:
        return None
    viable = feas_df[feas_df["oom"] == "no"].copy() if "oom" in feas_df.columns else feas_df
    tp = _throughput_col(viable)
    col = next((c for c in ["est_total_min_per_epoch", "est_min_per_epoch_30ep"]
                if c in viable.columns), None)
    if tp and col and not viable.empty:
        idx = viable[tp].idxmax()
        if not pd.isna(idx):
            return float(viable.loc[idx, col])
    return None


def _real_min_per_epoch(r) -> float | None:
    df = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
    if "epoch_time" in df.columns and df["epoch_time"].notna().any():
        return float(df["epoch_time"].mean()) / 60.0
    return None


def render_validate(ctx) -> object:
    """Compare-style validation: pick runs, see the feasibility prediction next to
    the real result for each (estimated vs real time, predicted vs real F1), as a
    table + grouped bars — the same look as the Compare section."""
    st.markdown("### Predicted vs actual")
    st.caption("Pick runs and compare what the feasibility **predicted** with what "
               "they actually did. Each run is matched to the feasibility report of "
               "its model: estimated vs real time/epoch and predicted vs real Val F1.")

    feas_csvs = _get_feasibility_csvs()
    if not feas_csvs:
        st.info("No feasibility reports yet — generate one in the **Measure** tab.")
        st.divider()
        return st.container()

    # Parse every report once: (env, model, meta, feas_df).
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
    sel = st.multiselect("Runs to compare against their prediction (max 8)",
                         list(labelled.keys()), default=default, max_selections=8)
    if not sel:
        st.info("Select at least one run.")
        st.divider()
        return st.container()

    # ── Table: predicted/estimated vs real per run ──────────────────────────────
    rows = []
    for lbl in sel:
        r = labelled[lbl]
        real_min = _real_min_per_epoch(r)
        df_r = _load_df(str(r.log_path), str(r.epoch_csv_path) if r.epoch_csv_path else None)
        real_f1 = _safe_max(df_r["val_f1"]) if "val_f1" in df_r.columns else float("nan")
        m, fdf = _feas_for(r.env, r.model)
        est_min = _est_min_per_epoch(fdf)
        pred_f1 = None
        if m:
            try:
                pred_f1 = float(m.get("prediction", {}).get("predicted_best_f1") or 0) or None
            except (TypeError, ValueError):
                pred_f1 = None
        err = ((real_min - est_min) / est_min * 100) if (real_min and est_min) else None
        rows.append({
            "Run": _short(lbl),
            "Model": (r.model or "—").replace("_patch16_224", ""),
            "Strategy": r.mode,
            "Est min/ep": round(est_min, 2) if est_min else None,
            "Real min/ep": round(real_min, 2) if real_min else None,
            "Time err %": round(err) if err is not None else None,
            "Pred F1": round(pred_f1, 3) if pred_f1 else None,
            "Real F1": round(float(real_f1), 3) if not pd.isna(real_f1) else None,
        })
    tdf = pd.DataFrame(rows).set_index("Run")
    st.dataframe(tdf, use_container_width=True)
    _dl_csv(tdf.reset_index(), "predicted_vs_real.csv", "Download comparison")

    # ── Grouped bars: estimated vs real min/epoch (Compare look) ────────────────
    plot = tdf.dropna(subset=["Est min/ep", "Real min/ep"])
    if not plot.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Estimated", x=list(plot.index), y=list(plot["Est min/ep"]),
                             marker_color="#94a3b8",
                             text=[f"{v:.2f}" for v in plot["Est min/ep"]], textposition="outside"))
        fig.add_trace(go.Bar(name="Real", x=list(plot.index), y=list(plot["Real min/ep"]),
                             marker_color="#3A536B",
                             text=[f"{v:.2f}" for v in plot["Real min/ep"]], textposition="outside"))
        fig.update_layout(**_base_layout(340, "Time per epoch — estimated vs real"),
                          barmode="group", yaxis_title="Minutes", xaxis_title="")
        _show(fig, "validate_time_bars")
        mean_err = float(plot.apply(
            lambda x: abs((x["Real min/ep"] - x["Est min/ep"]) / x["Est min/ep"] * 100),
            axis=1).mean())
        if mean_err <= 15:
            st.success(f"Predictions are accurate — mean |error| {mean_err:.0f}% in "
                       f"time/epoch across {len(plot)} run(s).")
        else:
            st.warning(f"Mean |error| {mean_err:.0f}% in time/epoch — the estimate is off "
                       "for some runs (often I/O on NFS, which the synthetic benchmark omits).")
    else:
        st.caption("No run has both an estimate (matching feasibility report) and real "
                   "timing — pick runs whose model has a feasibility report.")

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


