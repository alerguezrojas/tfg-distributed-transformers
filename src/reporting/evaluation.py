"""Statistics of the evaluation section: everything computed from one inspected epoch.

Pure NumPy through ``src.analysis``; no plotting here (``figures.py`` draws, ``render.py``
writes). The probabilities are those of the **validation** partition the monitor saved (the
whole partition if the run stored ``val_probs``, else the fixed monitor subset), so the numbers
are an internal-validation estimate, not a held-out test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src import analysis as an

DEFAULT_THRESHOLD = 0.5
SENSITIVITIES = (0.90, 0.95)


class NoInspection(ValueError):
    """The run has no usable inspection data for the requested epoch."""


@dataclass
class EvalInputs:
    """Arrays of one inspected epoch, ready for the analysis functions."""
    epoch: int
    task: str
    classes: list[str]
    probs: np.ndarray            # (N, C) float16 as stored
    targets: np.ndarray          # (N, C) uint8
    source: str                  # "full validation partition" | "monitor subset"
    threshold: float
    threshold_source: str
    monitor: an.MonitorSet
    epoch_data: an.EpochData


def choose_epoch(epochs: list[int], requested: int | None) -> int:
    """The requested epoch, or the last one inspected."""
    if not epochs:
        raise NoInspection("no epoch_<EEE>.npz in the inspection directory")
    if requested is None:
        return epochs[-1]
    if requested not in epochs:
        raise NoInspection(f"epoch {requested} was not inspected; available: {epochs}")
    return requested


def load_inputs(inspect_dir: Path, epoch: int | None) -> EvalInputs:
    """Read the monitor set and one epoch; choose full validation if it was stored."""
    epochs = an.list_epochs(inspect_dir)
    ep = choose_epoch(epochs, epoch)
    mon = an.load_monitor_set(inspect_dir)
    data = an.load_epoch(inspect_dir, ep)
    if data.val_probs is not None and data.val_targets is not None:
        probs, targets, source = data.val_probs, data.val_targets, "full validation partition"
    else:
        probs, targets, source = data.probs, mon.targets, "fixed monitor subset of validation"
    if data.threshold is not None:
        thr, thr_src = float(data.threshold), "validation threshold stored by the run"
    else:
        thr, thr_src = DEFAULT_THRESHOLD, "default 0.5 (the run stored no threshold)"
    if mon.task == "multiclass":
        thr, thr_src = float("nan"), "not applicable: the prediction is the argmax"
    return EvalInputs(epoch=ep, task=mon.task, classes=mon.classes, probs=probs, targets=targets,
                      source=source, threshold=thr, threshold_source=thr_src, monitor=mon,
                      epoch_data=data)


@dataclass
class Evaluation:
    """Results of the evaluation section."""
    inputs: EvalInputs
    n: int
    n_boot: int
    seed: int
    alpha: float
    bins: int
    metrics: dict[str, Any]
    curves: dict[str, Any]
    calibration: dict[str, Any]
    cal_stats: dict[str, Any]
    class_cal: dict[str, Any]
    adaptive: dict[str, Any]
    confusion: dict[str, Any]
    op: dict[str, Any]
    per_class: list[dict[str, Any]] = field(default_factory=list)


def evaluate(inp: EvalInputs, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
             bins: int = 15) -> Evaluation:
    """All the statistics of the evaluation section for one epoch."""
    p, y, task = inp.probs, inp.targets, inp.task
    # multiclass ignores the threshold; the analysis functions still want a number
    thr = DEFAULT_THRESHOLD if np.isnan(inp.threshold) else inp.threshold
    metrics = an.metrics_with_ci(p, y, task, threshold=thr, n_boot=n_boot, alpha=alpha,
                                 seed=seed, bins=bins)
    curves = an.per_class_curves(p, y, max_points=300)
    cal = an.calibration(p, y, task, bins=bins)
    stats = an.calibration_curve_stats(p, y, task)
    class_cal = an.per_class_calibration(p, y, task, bins=bins)
    adaptive = an.adaptive_ece(p, y, task, bins=bins)
    conf = an.confusion(p, y, task, threshold=thr)
    op = an.operating_points(p, y, task, threshold=thr, sensitivities=SENSITIVITIES)
    ev = Evaluation(inputs=inp, n=int(len(p)), n_boot=n_boot, seed=seed, alpha=alpha, bins=bins,
                    metrics=metrics, curves=curves, calibration=cal, cal_stats=stats,
                    class_cal=class_cal, adaptive=adaptive, confusion=conf, op=op)
    ev.per_class = _per_class_rows(ev)
    return ev


def _per_class_rows(ev: Evaluation) -> list[dict[str, Any]]:
    fixed = ev.op["fixed"]
    rows = []
    for c, name in enumerate(ev.inputs.classes):
        rows.append({
            "class": name,
            "n_pos": int(fixed["tp"][c] + fixed["fn"][c]),
            "prevalence": float(fixed["prevalence"][c]),
            "auc": float(ev.curves["auc"][c]),
            "ap": float(ev.curves["ap"][c]),
            "sensitivity": float(fixed["sensitivity"][c]),
            "specificity": float(fixed["specificity"][c]),
            "ppv": float(fixed["ppv"][c]),
            "npv": float(fixed["npv"][c]),
            "ece": float(ev.class_cal["ece"][c]),
            "slope": float(ev.cal_stats["per_class"][c]["slope"]),
            "intercept": float(ev.cal_stats["per_class"][c]["intercept"]),
        })
    return rows


def confusion_view(ev: Evaluation) -> dict[str, Any]:
    """Matrix, labels and a one-line reading guide, by task.

    Binary gets the usual 2 x 2 (the analysis returns the 1 x 1 co-activation of its single
    column, which says nothing); multi-label keeps the co-activation matrix and its note.
    """
    task, classes, conf = ev.inputs.task, ev.inputs.classes, ev.confusion
    if task == "binary":
        pc = conf["per_class"][0]
        m = np.array([[pc["tn"], pc["fp"]], [pc["fn"], pc["tp"]]], dtype=np.int64)
        return {"matrix": m, "rows": ["negative", "positive"], "cols": ["negative", "positive"],
                "row_title": "True", "col_title": "Predicted",
                "note": f"Binary task, threshold {ev.inputs.threshold:.3f}."}
    m = np.asarray(conf["matrix"])
    if task == "multiclass":
        return {"matrix": m, "rows": classes, "cols": classes, "row_title": "True class",
                "col_title": "Predicted class", "note": "Rows: true class; columns: argmax."}
    return {"matrix": m, "rows": classes, "cols": classes, "row_title": "Label present",
            "col_title": "Predicted positive",
            "note": ("Multi-label co-activation counts: a sample is counted in every (label, "
                     "prediction) pair it has; the diagonal is the true positives of each class.")}


def dynamics_inputs(inspect_dir: Path, epochs: list[int]):
    """(prob_of_true, correct) stacked over the inspected epochs: two (E, M) arrays."""
    mon = an.load_monitor_set(inspect_dir)
    pt, ok = [], []
    for e in epochs:
        d = an.load_epoch(inspect_dir, e)
        thr = DEFAULT_THRESHOLD if d.threshold is None else d.threshold
        pt.append(an.prob_of_true(d.probs, mon.targets, mon.task))
        ok.append(an.sample_correct(d.probs, mon.targets, mon.task, thr))
    return np.array(pt), np.array(ok)
