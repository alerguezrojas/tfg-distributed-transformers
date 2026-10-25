"""Curves sub-view: ROC / PR per class, metrics against the threshold, per-class table and
the evolution of the macro AUC / AP across epochs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.curves import per_class_curves
from src.analysis.thresholds import threshold_sweep
from src.web.tabs.inspector._common import InspectorCtx, _mtime, fmt, need_monitor
from src.web.ui.charts import _show

FULL, SUBSET = "Full validation set", "Monitored subset"


def arrays(c: InspectorCtx, epoch: int, source: str):
    """(probs, targets) of ``epoch`` for the chosen source, or None if it is unavailable."""
    d = c.epoch_data(epoch)
    if d is None:
        return None
    if source == FULL:
        if d.val_probs is None or d.val_targets is None:
            return None
        return d.val_probs, d.val_targets
    return d.probs, c.monitor.targets


@st.cache_data(show_spinner=False, max_entries=24)
def _curves(dir_str: str, epoch: int, mtime: float, source: str, _probs, _targets) -> dict:
    # Underscore arguments are not hashed: (dir, epoch, mtime, source) identifies the data.
    return per_class_curves(_probs, _targets, max_points=600)


@st.cache_data(show_spinner=False, max_entries=24)
def _sweep(dir_str: str, epoch: int, mtime: float, source: str, task: str, _probs,
           _targets) -> dict:
    return threshold_sweep(_probs, _targets, task)


def source_choice(c: InspectorCtx, key: str) -> str | None:
    """Full-validation arrays when the selected epoch has them (the default), else the subset."""
    d = c.epoch_data()
    if d is not None and d.val_probs is not None and d.val_targets is not None:
        n = len(d.val_probs)
        return st.radio("Data", [FULL, SUBSET], horizontal=True, key=key,
                        format_func=lambda s: f"{s} ({n:,} samples)" if s == FULL
                        else f"{s} ({len(d.probs):,} samples)")
    st.caption(f"Computed on the **monitored subset** ({len(d.probs):,} samples): this run did not "
               "save the full validation probabilities (`monitor.full_val_max`).")
    return SUBSET


def render(c: InspectorCtx) -> None:
    if not need_monitor(c):
        return
    source = source_choice(c, "insp_curves_source")
    probs, targets = arrays(c, c.epoch, source)
    mt = _mtime(c.inspect_dir / f"epoch_{c.epoch:03d}.npz")
    out = _curves(str(c.inspect_dir), c.epoch, mt, source, probs, targets)

    k = st.columns(4)
    k[0].metric("Macro AUC", fmt(out["macro"]["auc"]))
    k[1].metric("Micro AUC", fmt(out["micro"]["roc"]["auc"]))
    k[2].metric("Macro AP", fmt(out["macro"]["ap"]))
    k[3].metric("Micro AP", fmt(out["micro"]["pr"]["ap"]))
    if c.task == "multiclass":
        st.caption("Multiclass: one-vs-rest curves per class.")

    shown = st.multiselect("Classes", c.classes, default=c.classes[:5], key="insp_curves_cls")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                                 line={"dash": "dot", "color": "#9AA0A6"}))
        _add_curve(fig, out["micro"]["roc"], "fpr", "tpr", f"micro (AUC {fmt(out['micro']['roc']['auc'])})",
                   dash="dash")
        for name in shown:
            i = c.classes.index(name)
            r = out["classes"][i]["roc"]
            _add_curve(fig, r, "fpr", "tpr", f"{name} ({fmt(r['auc'])})")
        fig.update_layout(height=380, title={"text": "ROC"}, xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        _show(fig, "insp_roc")
    with right:
        fig = go.Figure()
        _add_curve(fig, out["micro"]["pr"], "recall", "precision",
                   f"micro (AP {fmt(out['micro']['pr']['ap'])})", dash="dash")
        for name in shown:
            i = c.classes.index(name)
            r = out["classes"][i]["pr"]
            _add_curve(fig, r, "recall", "precision", f"{name} ({fmt(r['ap'])})")
        fig.update_layout(height=380, title={"text": "Precision-recall"}, xaxis_title="Recall",
                          yaxis_title="Precision", yaxis_range=[0, 1.02])
        _show(fig, "insp_pr")

    st.markdown("---")
    sweep = _sweep(str(c.inspect_dir), c.epoch, mt, source, c.task, probs, targets)
    thr = _threshold_section(c, sweep)
    _class_table(c, out, probs, targets, thr)
    _evolution(c, source)


def _add_curve(fig: go.Figure, curve: dict, x: str, y: str, name: str, dash: str | None = None):
    if len(curve[x]):
        fig.add_trace(go.Scatter(x=curve[x], y=curve[y], mode="lines", name=name,
                                 line={"width": 1.6, "dash": dash} if dash else {"width": 1.6}))


def _threshold_section(c: InspectorCtx, sweep: dict) -> float | None:
    st.markdown("##### Metrics against the decision threshold")
    if c.task == "multiclass":
        st.caption("Multiclass has no decision threshold (the prediction is the argmax): "
                   "these are the metrics at the argmax decision.")
        return None
    fig = go.Figure()
    for key, name in (("f1", "F1 (macro)"), ("precision", "Precision"),
                      ("sensitivity", "Sensitivity (recall)"), ("specificity", "Specificity")):
        fig.add_trace(go.Scatter(x=sweep["thresholds"], y=sweep[key], mode="lines+markers",
                                 name=name, line={"width": 1.8}, marker={"size": 4}))
    best = sweep["best"]
    fig.add_vline(x=best["threshold"], line_dash="dash", line_color="#2CA02C",
                  annotation_text=f"best F1 {best['f1']:.3f} @ {best['threshold']:.2f}")
    fig.update_layout(height=340, xaxis_title="Threshold", yaxis_title="Value",
                      yaxis_range=[0, 1.02], hovermode="x unified")
    _show(fig, "insp_threshold")
    d = c.epoch_data()
    default = float(round(d.threshold, 2)) if d.threshold is not None else best["threshold"]
    return st.slider("Threshold for the table below", 0.05, 0.95, min(max(default, 0.05), 0.95),
                     0.05, key="insp_curves_thr",
                     help="Defaults to the epoch's validation-optimal threshold.")


def _class_table(c: InspectorCtx, out: dict, probs, targets, thr: float | None) -> None:
    grid = None if thr is None else [thr]
    at = threshold_sweep(probs, targets, c.task, grid=grid)["per_class"]
    support = (np.asarray(targets) > 0).sum(axis=0)
    df = pd.DataFrame({
        "Class": c.classes, "Support": support, "AUC": out["auc"], "AP": out["ap"],
        "F1": at["f1"][0], "Sensitivity": at["sensitivity"][0],
        "Specificity": at["specificity"][0], "Precision": at["precision"][0],
    })
    label = "argmax" if thr is None else f"threshold {thr:.2f}"
    st.markdown(f"##### Per class ({label})")
    st.dataframe(df.round(3), hide_index=True, width="stretch")


def _evolution(c: InspectorCtx, source: str) -> None:
    rows = []
    for e in c.epochs:
        if e > c.epoch:
            break
        a = arrays(c, e, source)
        if a is None:
            continue
        o = _curves(str(c.inspect_dir), e, _mtime(c.inspect_dir / f"epoch_{e:03d}.npz"),
                    source, *a)
        rows.append({"epoch": e, "Macro AUC": o["macro"]["auc"], "Macro AP": o["macro"]["ap"]})
    st.markdown("##### Evolution")
    if len(rows) < 2:
        st.caption("Needs at least two inspected epochs.")
        return
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for col in ("Macro AUC", "Macro AP"):
        fig.add_trace(go.Scatter(x=df["epoch"], y=df[col], mode="lines+markers", name=col))
    fig.update_layout(height=300, xaxis_title="epoch", yaxis_title="score", hovermode="x unified")
    fig.update_xaxes(dtick=1)
    _show(fig, "insp_curves_evolution")
