"""Clinical sub-view: choose the operating point of a class and read what it means.

Sensitivity, specificity, PPV, NPV, F1 and the 2x2 table at a threshold chosen by hand, by a
target sensitivity (rule-out), by Youden's J or taken from the run's own validation, with
bootstrap intervals, an editable prevalence for the predictive values, and the ROC curve with
the point marked. Multiclass is one-vs-rest on the probability of the chosen class.
"""
from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from src.web.inspector_stats import (
    CRITERIA, DEFAULT_SEED, N_BOOT_CHOICES, POINT_METRICS, ci_note, clinical_report,
    confusion_2x2, point_frame,
)
from src.web.tabs.inspector._common import InspectorCtx, _mtime, fmt, need_monitor
from src.web.tabs.inspector.curves import arrays, source_choice
from src.web.ui.charts import _show

SENS_TARGETS = (0.80, 0.90, 0.95, 0.99)
_LABELS = {"sensitivity": "Sensitivity", "specificity": "Specificity", "ppv": "PPV",
           "npv": "NPV", "f1": "F1", "accuracy": "Accuracy"}
_HELP = {"sensitivity": "Share of the real positives the model catches (recall).",
         "specificity": "Share of the real negatives it clears.",
         "ppv": "Of those called positive, the share that really are. Depends on the prevalence.",
         "npv": "Of those called negative, the share that really are. Depends on the prevalence.",
         "f1": "Harmonic mean of PPV and sensitivity.", "accuracy": "Share of correct calls."}


@st.cache_data(show_spinner="Computing the operating point…", max_entries=32)
def _report(dir_str: str, epoch: int, mtime: float, source: str, cls: int, criterion: str,
            thr: float, target: float, prev: float | None, n_boot: int, _probs, _targets) -> dict:
    # Underscore arguments are not hashed: (dir, epoch, mtime, source) identifies the data.
    return clinical_report(_probs, _targets, cls, criterion, thr, target, prev, n_boot,
                           DEFAULT_SEED)


def _controls(c: InspectorCtx, targets, names: list[str]):
    d = c.epoch_data()
    cls = names.index(st.selectbox("Class", names, key="insp_clin_cls")) if len(names) > 1 else 0
    options = [k for k in CRITERIA if k != "validation" or
               (d is not None and d.threshold is not None and c.task != "multiclass")]
    criterion = st.radio("Operating point", options, horizontal=True, key="insp_clin_crit",
                         format_func=lambda k: CRITERIA[k])
    thr, target = 0.5, 0.90
    if criterion == "fixed":
        thr = st.slider("Threshold (positive when p > threshold)", 0.01, 0.99, 0.5, 0.01,
                        key="insp_clin_thr")
    elif criterion == "validation":
        thr = float(d.threshold)
        st.caption(f"Threshold {thr:.3f}: the validation-optimal one recorded by the run "
                   "(chosen on the full validation set).")
    elif criterion == "sensitivity":
        target = st.select_slider("Sensitivity at least", SENS_TARGETS, value=0.90,
                                  key="insp_clin_target")
    observed = float((targets[:, cls] > 0).mean())
    a, b, e = st.columns([1.2, 1.2, 1.2], vertical_alignment="bottom")
    own = a.toggle("Set the prevalence", value=False, key="insp_clin_prev_on",
                   help="PPV and NPV depend on how common the condition is. By default they use "
                        "the prevalence of this data; set the one of the population you "
                        "will deploy on.")
    prev = None
    if own:
        pct = b.number_input("Prevalence (%)", 0.1, 99.9, min(max(100 * observed, 0.1), 99.9),
                             0.1, key="insp_clin_prev")
        prev = pct / 100.0
    n_boot = e.selectbox("Interval resamples", (0, *N_BOOT_CHOICES), index=2, key="insp_clin_n",
                         format_func=lambda n: "no intervals" if n == 0 else f"{n:,}")
    return cls, criterion, thr, target, prev, n_boot


def render(c: InspectorCtx) -> None:
    if not need_monitor(c):
        return
    source = source_choice(c, "insp_clin_source")
    probs, targets = arrays(c, c.epoch, source)
    names = c.classes or [f"class {i}" for i in range(targets.shape[1])]
    if c.task == "multiclass":
        st.caption("Multiclass: one class against the rest, thresholding the probability of the "
                   "chosen class (not the argmax).")
    cls, criterion, thr, target, prev, n_boot = _controls(c, targets, names)
    mt = _mtime(c.inspect_dir / f"epoch_{c.epoch:03d}.npz")
    r = _report(str(c.inspect_dir), c.epoch, mt, source, cls, criterion, thr, target, prev,
                n_boot, probs, targets)
    if not r["ok"]:
        st.info(r["reason"])
        return

    st.markdown(f"##### {names[cls]} at threshold {r['threshold']:.3f}")
    cols = st.columns(len(POINT_METRICS))
    for col, m in zip(cols, POINT_METRICS):
        v = r["metrics"][m]
        finite = not (math.isnan(v["lo"]) or math.isnan(v["hi"]))
        col.metric(_LABELS[m], fmt(v["estimate"]), help=_HELP[m],
                   delta=f"[{v['lo']:.3f}, {v['hi']:.3f}]" if finite else None, delta_color="off")
    obs, used = r["prevalence"]["observed"], r["prevalence"]["used"]
    note = (f"Prevalence in this data: {100 * obs:.1f}% ({r['n']:,} samples). PPV and NPV use "
            + (f"{100 * used:.1f}% (set by you)." if prev is not None else "that prevalence."))
    if r["n_boot"]:
        note += " " + ci_note(r["n_boot"], seed=r["seed"])
        if criterion in ("sensitivity", "youden"):
            note += (" The threshold was chosen on these same data: the intervals are "
                     "conditional on it and ignore the uncertainty of that choice.")
    st.caption(note)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**2×2 table**")
        st.dataframe(confusion_2x2(r["counts"]), width="stretch")
        st.dataframe(point_frame(r), hide_index=True, width="stretch")
        if criterion == "sensitivity":
            got = r["metrics"]["sensitivity"]["estimate"]
            st.caption(f"Asked for sensitivity ≥ {target:.2f}; the highest threshold that "
                       f"reaches it gives {got:.3f} on these data.")
    with right:
        roc = r["roc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                                 line={"dash": "dot", "color": "#9AA0A6"}))
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                 name=f"ROC (AUC {fmt(roc['auc'])})", line={"width": 1.8}))
        fig.add_trace(go.Scatter(x=[r["point"][0]], y=[r["point"][1]], mode="markers",
                                 name=f"threshold {r['threshold']:.3f}",
                                 marker={"size": 12, "symbol": "diamond", "color": "#D62728",
                                         "line": {"width": 1, "color": "white"}}))
        fig.update_layout(height=380, title={"text": f"ROC · {names[cls]}"},
                          xaxis_title="1 - specificity", yaxis_title="Sensitivity")
        _show(fig, "insp_clin_roc")
    st.caption("Fix the threshold on validation data and apply it unchanged to test: choosing it "
               "on the same data it is evaluated on makes the figures optimistic.")
