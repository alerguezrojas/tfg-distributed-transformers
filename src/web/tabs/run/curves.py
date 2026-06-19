"""Run results — curves view."""
from __future__ import annotations


import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.eval_parser import parse_eval_csv
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _metric_fig, _show)
from src.web.ui.context import DashboardContext
from src.web.ui.helpers import (_dur_str, _load_df, _safe_max, _safe_val_at_best)


def _test_callout(ctx: DashboardContext) -> None:
    """Held-out TEST-set results (from scripts/eval.py), shown above the tabs.

    Every metric elsewhere in the dashboard is *validation*; this is the honest
    final number on the test split that training never touches. Appears only
    when a test_*.csv sits in the run's folder."""
    run = ctx.run
    if run is None or not getattr(run, "test_csv_paths", None):
        return
    with st.container(border=True):
        st.markdown("#### Held-out test set")
        st.caption(
            "Evaluated on the **test** split (never seen in training) with "
            "`scripts/eval.py`. Every other figure here is validation — this is "
            "the final generalization number."
        )
        for p in run.test_csv_paths:
            pcdf, agg = parse_eval_csv(p)
            if not agg and pcdf.empty:
                continue
            st.markdown(f"**{p.name}**")
            cols = st.columns(4)
            if "f1_t05" in agg:
                cols[0].metric("Macro F1 (t=0.50)", f"{float(agg['f1_t05']):.4f}")
            if "f1_opt" in agg:
                _t = agg.get("threshold")
                cols[1].metric("Macro F1 (optimal)", f"{float(agg['f1_opt']):.4f}",
                               delta=f"t={float(_t):.2f}" if _t is not None else None,
                               delta_color="off")
            if "accuracy" in agg:
                cols[2].metric("Accuracy", f"{float(agg['accuracy']):.4f}")
            if "loss" in agg:
                cols[3].metric("BCE loss", f"{float(agg['loss']):.4f}")
            if not pcdf.empty and "class_name" in pcdf.columns:
                zeros = pcdf[pcdf["f1"] == 0]["class_name"].tolist()
                if zeros:
                    st.caption(f"Classes with F1=0 on test ({len(zeros)}): "
                               + ", ".join(zeros[:6]) + ("…" if len(zeros) > 6 else ""))
                with st.expander("Per-class test metrics"):
                    st.dataframe(pcdf, use_container_width=True, hide_index=True)
                    _dl_csv(pcdf, f"{p.stem}.csv", "Download")


def _curves(ctx: DashboardContext) -> None:
    runs = ctx.runs
    selected_run = ctx.selected_run
    run = ctx.run
    refresh_interval = ctx.refresh_interval
    if selected_run is None:
        st.info("Select a run in the sidebar.")
    else:
        df = _load_df(
            str(run.log_path),
            str(run.epoch_csv_path) if run.epoch_csv_path else None,
        )

        if df.empty:
            st.error("Could not parse any epoch from the selected run.")
        else:
            n_epochs = len(df)
            best_f1 = _safe_max(df["val_f1"]) if "val_f1" in df.columns else float("nan")
            best_ep_v = _safe_val_at_best(df, "val_f1", "epoch")
            best_epoch = int(best_ep_v) if best_ep_v is not None else "—"
            best_thresh_f1 = (
                _safe_max(df["f1_at_threshold"])
                if "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any()
                else None
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Epochs", n_epochs)
            c2.metric("Best Val F1", f"{best_f1:.4f}" if not pd.isna(best_f1) else "—")
            c3.metric("Best epoch", best_epoch)
            if best_thresh_f1 is not None:
                c4.metric("F1 @ optimal threshold", f"{best_thresh_f1:.4f}")
            if "epoch_time" in df.columns and df["epoch_time"].notna().any():
                c5.metric("Total duration", _dur_str(df["epoch_time"].sum()))

            src = "epoch_metrics CSV" if (run.epoch_csv_path and run.epoch_csv_path.exists()) else "log file"
            st.caption(f"Source: {src}")

            # One-line verdict: best epoch, overfitting gap, val-loss divergence.
            if not pd.isna(best_f1) and best_ep_v is not None and "train_f1" in df.columns:
                _tr = df.loc[df["epoch"] == best_ep_v, "train_f1"]
                bits = [f"Best Val F1 **{best_f1:.3f}** at epoch {int(best_ep_v)}"]
                if not _tr.empty:
                    gap = float(_tr.iloc[0]) - float(best_f1)
                    bits.append(f"train–val gap {gap:+.2f}" + (" → overfitting" if gap > 0.1 else ""))
                if "val_loss" in df.columns and df["val_loss"].notna().sum() > 2:
                    vl = df.sort_values("epoch")["val_loss"].dropna()
                    if len(vl) > 2 and vl.iloc[-1] > vl.min() * 1.15:
                        bits.append("val loss diverges after the best epoch")
                (st.warning if "overfitting" in " ".join(bits) else st.info)(" · ".join(bits))

            extra_thresh: list = []
            if "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any():
                extra_thresh = [go.Scatter(
                    x=df["epoch"], y=df["f1_at_threshold"],
                    name="F1 @ optimal threshold", mode="lines",
                    line=dict(color=COLORS[2], width=2, dash="dot"),
                )]

            c1, c2 = st.columns(2)
            with c1:
                _show(_metric_fig(df, "train_f1", "val_f1", "F1 (macro)", "F1", extra_traces=extra_thresh), "f1")
                _show(_metric_fig(df, "train_loss", "val_loss", "Loss (BCE)", "Loss"), "loss")
            with c2:
                _show(_metric_fig(df, "train_acc", "val_acc", "Accuracy", "Accuracy",
                                  color_train=COLORS[4], color_val=COLORS[5]), "accuracy")
                _show(_metric_fig(df, "val_prec", "val_rec", "Precision & Recall (val)",
                                  "Score", color_train=COLORS[2], color_val=COLORS[3]), "prec_rec")

            if "epoch_time" in df.columns and df["epoch_time"].notna().any():
                et = df[["epoch", "epoch_time"]].dropna()
                fig_et = go.Figure(go.Bar(x=et["epoch"], y=et["epoch_time"] / 60,
                                          marker_color=COLORS[0], opacity=0.8))
                fig_et.update_layout(**_base_layout(240, "Time per epoch (min)"),
                                     xaxis_title="Epoch", yaxis_title="Minutes")
                _show(fig_et, "epoch_time")

            has_energy = "energy_eval_wh" in df.columns and df["energy_eval_wh"].notna().any()
            if has_energy:
                st.markdown("#### Energy consumption")
                e1, e2 = st.columns(2)
                with e1:
                    rows_e = []
                    for _, row in df.iterrows():
                        if pd.notna(row.get("energy_eval_wh")):
                            entry = {"epoch": row["epoch"], "Eval (Wh)": row["energy_eval_wh"]}
                            if pd.notna(row.get("energy_train_j")):
                                entry["Train (Wh)"] = row["energy_train_j"] / 3600
                            rows_e.append(entry)
                    if rows_e:
                        df_e = pd.DataFrame(rows_e)
                        fig_e = go.Figure()
                        if "Train (Wh)" in df_e.columns:
                            fig_e.add_trace(go.Bar(x=df_e["epoch"], y=df_e["Train (Wh)"],
                                                   name="Train", marker_color=COLORS[0], opacity=0.85))
                        fig_e.add_trace(go.Bar(x=df_e["epoch"], y=df_e["Eval (Wh)"],
                                               name="Eval", marker_color=COLORS[1], opacity=0.85))
                        fig_e.update_layout(**_base_layout(260, "Energy per epoch (Wh)"),
                                            barmode="group", xaxis_title="Epoch", yaxis_title="Wh")
                        _show(fig_e, "energy")
                with e2:
                    power_cols = []
                    if "power_eval_w" in df.columns and df["power_eval_w"].notna().any():
                        power_cols.append(("Eval power (W)", "power_eval_w", COLORS[1]))
                    if "power_train_w" in df.columns and df["power_train_w"].notna().any():
                        power_cols.append(("Train power (W)", "power_train_w", COLORS[0]))
                    if power_cols:
                        fig_p = go.Figure()
                        for name_p, col_p, color_p in power_cols:
                            fig_p.add_trace(go.Scatter(x=df["epoch"], y=df[col_p],
                                                       name=name_p, mode="lines+markers",
                                                       line=dict(color=color_p, width=2)))
                        fig_p.update_layout(**_base_layout(260, "Average GPU power per epoch (W)"),
                                            xaxis_title="Epoch", yaxis_title="Watts")
                        _show(fig_p, "gpu_power")

                total_eval_wh = df["energy_eval_wh"].sum()
                total_train_wh = (df["energy_train_j"].sum() / 3600
                                  if "energy_train_j" in df.columns and df["energy_train_j"].notna().any() else 0)
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("Total eval energy", f"{total_eval_wh:.1f} Wh")
                if total_train_wh > 0:
                    ec2.metric("Total train energy", f"{total_train_wh:.1f} Wh")
                    ec3.metric("Total energy", f"{total_eval_wh + total_train_wh:.1f} Wh")

            _dl_csv(df, "epoch_metrics.csv", "Download epoch_metrics.csv")

            with st.expander("Full epochs table"):
                st.dataframe(df.set_index("epoch"), use_container_width=True)



