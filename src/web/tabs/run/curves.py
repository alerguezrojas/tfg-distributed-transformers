"""Run results — curves view."""
from __future__ import annotations


import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.web.eval_parser import parse_eval_csv
from src.web.ui.charts import (COLORS, _base_layout, _dl_csv, _metric_fig, _show)
from src.web.ui.context import DashboardContext
from src.web.run_meta import AUC, F1, QWK, MetricInfo, has_values
from src.web.ui.helpers import (_best_metric, _dur_str, _load_df, _loss_label, _run_metric,
                                 _safe_max)

_TRAIN_WH = "Train (Wh)"


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
            cols = st.columns(5)
            if "f1_t05" in agg:
                cols[0].metric("Macro F1 (t=0.50)", f"{float(agg['f1_t05']):.4f}")
            if "f1_opt" in agg:
                _t = agg.get("threshold")
                # Older eval CSVs carry no source: their threshold was tuned on test.
                _tuned_on_val = agg.get("threshold_source") in ("checkpoint", "val")
                cols[1].metric("Macro F1 (val-tuned t)" if _tuned_on_val
                               else "Macro F1 (test-tuned t)",
                               f"{float(agg['f1_opt']):.4f}",
                               delta=f"t={float(_t):.2f}" if _t is not None else None,
                               delta_color="off",
                               help=None if _tuned_on_val else
                               "Threshold tuned on the test set itself: optimistic.")
            if "accuracy" in agg:
                cols[2].metric("Accuracy", f"{float(agg['accuracy']):.4f}")
            if "auc" in agg:
                cols[3].metric("AUC (macro)", f"{float(agg['auc']):.4f}")
            if "loss" in agg:
                cols[4].metric("Loss", f"{float(agg['loss']):.4f}")
            if not pcdf.empty and "class_name" in pcdf.columns:
                zeros = pcdf[pcdf["f1"] == 0]["class_name"].tolist()
                if zeros:
                    st.caption(f"Classes with F1=0 on test ({len(zeros)}): "
                               + ", ".join(zeros[:6]) + ("…" if len(zeros) > 6 else ""))
                with st.expander("Per-class test metrics"):
                    st.dataframe(pcdf, width="stretch", hide_index=True)
                    _dl_csv(pcdf, f"{p.stem}.csv", "Download")


def _metric_curve(df: pd.DataFrame, metric: MetricInfo, best: tuple[float, int] | None) -> go.Figure:
    """Validation curve of ``metric`` (plus its train side for F1 and the F1 at the optimal
    threshold), with the best epoch starred."""
    extra: list = []
    if metric.key == "f1" and "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any():
        extra.append(go.Scatter(
            x=df["epoch"], y=df["f1_at_threshold"], name="F1 @ optimal threshold",
            mode="lines", line={"color": COLORS[2], "width": 2, "dash": "dot"}))
    if best:
        extra.append(go.Scatter(
            x=[best[1]], y=[best[0]], name=f"Best (epoch {best[1]})", mode="markers",
            marker={"color": COLORS[3], "size": 11, "symbol": "star"}))
    return _metric_fig(df, "train_f1" if metric.key == "f1" else "", metric.column,
                       metric.long_name, metric.name, extra_traces=extra,
                       name_val="Val" if metric.key == "f1" else f"Val {metric.name}")


def _curves(ctx: DashboardContext) -> None:
    selected_run = ctx.selected_run
    run = ctx.run
    if selected_run is None:
        st.info("Select a run in the sidebar.")
        return
    df = _load_df(
        str(run.log_path),
        str(run.epoch_csv_path) if run.epoch_csv_path else None,
    )
    if df.empty:
        st.error("Could not parse any epoch from the selected run.")
        return

    # The run's own main metric (QWK for PANDA, AUC for MedMNIST, F1 for BigEarthNet) leads;
    # the others are secondary and appear only when the epoch table has them.
    metric = _run_metric(run, df)
    main_best = _best_metric(df, metric.column)
    f1_best = _best_metric(df, "val_f1")
    auc_best = _best_metric(df, "val_auc")
    best_thresh_f1 = (_safe_max(df["f1_at_threshold"])
                      if "f1_at_threshold" in df.columns and df["f1_at_threshold"].notna().any()
                      else None)

    cards: list[tuple[str, str, str | None]] = [
        ("Epochs", str(len(df)), None),
        (f"Best Val {metric.name}", f"{main_best[0]:.4f}" if main_best else "—", None),
        ("Best epoch", str(main_best[1]) if main_best else "—", None),
    ]
    if metric.key != "f1" and f1_best:
        cards.append(("Best Val F1", f"{f1_best[0]:.4f}", f"epoch {f1_best[1]}"))
    if metric.key != "auc" and auc_best:
        cards.append(("Best Val AUC", f"{auc_best[0]:.4f}", f"epoch {auc_best[1]}"))
    if metric.key == "f1" and best_thresh_f1 is not None:
        cards.append(("F1 @ optimal threshold", f"{best_thresh_f1:.4f}", None))
    if "epoch_time" in df.columns and df["epoch_time"].notna().any():
        cards.append(("Total duration", _dur_str(df["epoch_time"].sum()), None))
    for col, (label, value, delta) in zip(st.columns(len(cards)), cards):
        col.metric(label, value, delta=delta, delta_color="off")

    src = "epoch_metrics CSV" if (run.epoch_csv_path and run.epoch_csv_path.exists()) else "log file"
    st.caption(f"Source: {src} · {run.dataset} · main metric: {metric.long_name} "
               f"· loss: {_loss_label(run)}")

    # One-line verdict: best epoch, overfitting gap (F1, the only train-side metric),
    # val-loss divergence.
    if main_best:
        bits = [f"Best Val {metric.name} **{main_best[0]:.3f}** at epoch {main_best[1]}"]
        if f1_best and "train_f1" in df.columns:
            _tr = df.loc[df["epoch"] == f1_best[1], "train_f1"]
            if not _tr.empty:
                gap = float(_tr.iloc[0]) - f1_best[0]
                bits.append(f"train–val F1 gap {gap:+.2f}" + (" → overfitting" if gap > 0.1 else ""))
        if "val_loss" in df.columns and df["val_loss"].notna().sum() > 2:
            vl = df.sort_values("epoch")["val_loss"].dropna()
            if len(vl) > 2 and vl.iloc[-1] > vl.min() * 1.15:
                bits.append("val loss diverges after the best epoch")
        # Yellow when the message flags a problem (overfitting gap OR a
        # diverging val loss); blue otherwise. Keeps colour and text in sync.
        _text = " · ".join(bits)
        _warn = "overfitting" in _text or "diverges" in _text
        (st.warning if _warn else st.info)(_text)

    # Charts in two columns: main metric, loss, then the secondary ones that exist.
    figs: list[tuple[go.Figure, str]] = [
        (_metric_curve(df, metric, main_best), "metric"),
        (_metric_fig(df, "train_loss", "val_loss", f"Loss ({_loss_label(run)})", "Loss"), "loss"),
    ]
    if metric.key != "f1":
        figs.append((_metric_curve(df, F1, f1_best), "f1"))
    figs += [
        (_metric_fig(df, "train_acc", "val_acc", "Accuracy", "Accuracy",
                     color_train=COLORS[4], color_val=COLORS[5]), "accuracy"),
        (_metric_fig(df, "val_prec", "val_rec", "Precision & Recall (val)",
                     "Score", color_train=COLORS[2], color_val=COLORS[3],
                     name_train="Precision", name_val="Recall"), "prec_rec"),
    ]
    for other in (AUC, QWK):
        if other.key != metric.key and has_values(df, other.column):
            figs.append((_metric_curve(df, other, _best_metric(df, other.column)), other.key))
    if "epoch_time" in df.columns and df["epoch_time"].notna().any():
        et = df[["epoch", "epoch_time"]].dropna()
        fig_et = go.Figure(go.Bar(x=et["epoch"], y=et["epoch_time"] / 60,
                                  marker_color=COLORS[0], opacity=0.8))
        fig_et.update_layout(**_base_layout(320, "Time per epoch (min)"),
                             xaxis_title="Epoch", yaxis_title="Minutes")
        figs.append((fig_et, "epoch_time"))
    cols = st.columns(2)
    for i, (fig, key) in enumerate(figs):
        with cols[i % 2]:
            _show(fig, key)

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
                        entry[_TRAIN_WH] = row["energy_train_j"] / 3600
                    rows_e.append(entry)
            if rows_e:
                df_e = pd.DataFrame(rows_e)
                fig_e = go.Figure()
                if _TRAIN_WH in df_e.columns:
                    fig_e.add_trace(go.Bar(x=df_e["epoch"], y=df_e[_TRAIN_WH],
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
                                               line={"color": color_p, "width": 2}))
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
        st.dataframe(df.set_index("epoch"), width="stretch")



