"""Self-contained HTML report of one run (``paravit report``).

Reads ``events_<ts>.jsonl`` and, if present, the ``inspect_<ts>/`` directory next to it; never
trains, never touches the GPU. Everything is embedded in one file (plotly.js once, images as
base64), so it can be sent by e-mail. Sections degrade with a message naming the flag that
brings the missing data.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src import analysis as an
from src.analysis._common import check_pair, predict
from src.reporting import figures as fg
from src.reporting import tripod
from src.reporting.data import RunData, load_run, resolve_run
from src.reporting.evaluation import (Evaluation, EvalInputs, NoInspection, confusion_view,
                                      dynamics_inputs, evaluate, load_inputs)
from src.reporting.render import (Raw, duration, esc, kv_table, note, num, page, png_data_uri,
                                  section, table, unavailable)

REPORT_VERSION = 1
CAM_FLAG = ("Train with --monitor (monitor.enabled: true); Grad-CAM needs monitor.gradcam > 0 "
            "and a model with a compatible layer.")
NEED_MONITOR = "Train with --monitor (monitor.enabled: true) to save the inspection data."
METRIC_NAMES = {
    "auc": "AUC (macro, one-vs-rest)", "ap": "Average precision (macro)",
    "f1": "F1 (macro, at threshold)", "sensitivity": "Sensitivity (macro, at threshold)",
    "specificity": "Specificity (macro, at threshold)", "accuracy": "Accuracy",
    "ece": "ECE (equal-width bins)", "brier": "Brier score",
}
_DIV_END = "</div>"
SECTIONS = [("sec-summary", "Summary"), ("sec-training", "Training"),
            ("sec-evaluation", "Evaluation"), ("sec-dynamics", "Training dynamics"),
            ("sec-tripod", "TRIPOD+AI checklist")]


def _scroll(inner: Any) -> Raw:
    """Horizontally scrollable wrapper (wide tables)."""
    return Raw(f"<div class=\"scroll\">{inner}{_DIV_END}")


def _grid2(*parts: Any) -> Raw:
    """Two-column grid (one column on narrow screens)."""
    return Raw("<div class=\"grid2\">" + "".join(esc(p) for p in parts) + _DIV_END)


class _Page:
    """Figure registry: gives every plotly div a deterministic id."""

    def __init__(self) -> None:
        self.n = 0

    def fig(self, figure) -> Raw:
        self.n += 1
        return Raw(f"<div class=\"fig\">{fg.to_div(figure, f'fig{self.n}')}</div>")


def epoch_rows(run: RunData) -> list[dict[str, Any]]:
    """One flat row per finished epoch: ``epoch``, ``train_<m>``, ``val_<m>``, time, best."""
    rows = []
    for ev in run.epoch_ends:
        row: dict[str, Any] = {"epoch": ev.get("epoch")}
        for split in ("train", "val"):
            for k, v in (ev.get(split) or {}).items():
                row[f"{split}_{k}"] = v
        for k in ("epoch_time_s", "best", "is_best"):
            row[k] = ev.get(k)
        rows.append(row)
    return rows


# ── 1. Summary ─────────────────────────────────────────────────────────────────

def _system_frames(run: RunData):
    from src.web.inspector_data import system_frames
    return system_frames(run.systems)


def _energy(gpu_df) -> tuple[float | None, int]:
    """(total Wh over the GPUs, number of GPUs counted)."""
    from src.web.inspector_data import energy_wh
    per_gpu = energy_wh(gpu_df)
    return (sum(per_gpu.values()), len(per_gpu)) if per_gpu else (None, 0)


def _status(run: RunData) -> str:
    if run.run_end:
        status = str(run.run_end.get("status", "completed"))
        err = run.run_end.get("error")
        extra = f" ({err})" if err else ""
        if run.early_stop:
            extra += f", early stop at epoch {run.early_stop.get('epoch')}"
        return status + extra
    return "no run_end event (still running, or the process died)"


def _best(run: RunData, rows: list[dict[str, Any]]) -> str:
    metric = (run.run_start or {}).get("select_metric") or "metric"
    best = (run.run_end or {}).get("best")
    if best is None and run.epoch_ends:
        best = run.epoch_ends[-1].get("best")
    if best is None:
        return "n/a"
    at = [r["epoch"] for r in rows if r.get("is_best")]
    where = f" at epoch {at[-1]}" if at else ""
    return f"{metric} = {num(best, 4)}{where}"


def _config_text(cfg: dict[str, Any]) -> str:
    try:
        import yaml
        return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    except Exception:
        return json.dumps(cfg, indent=2, ensure_ascii=False, default=str)


def _energy_text(gpu_df) -> str:
    wh, n_gpu = _energy(gpu_df)
    if wh is None:
        return ("n/a (no system events with GPU power: keep --trace other than off and "
                "monitor.system_interval_s > 0, on a machine with NVML)")
    return (f"about {wh:.2f} Wh over {n_gpu} GPU(s): trapezoidal integral of the sampled "
            "GPU power, GPU only; not a metered value")


def _duration_text(run: RunData) -> str:
    total = (run.run_end or {}).get("total_time_s")
    if total is None and run.first_time is not None and run.last_time is not None:
        return duration(run.last_time - run.first_time) + " (first to last event; no run_end)"
    return duration(total)


def _gpu_rows(run: RunData, gpu_df) -> list[list[Any]]:
    """(index, name, uuid) per GPU; the uuid comes from the system events, not from the frame."""
    if gpu_df.empty:
        return []
    seen: dict[int, str] = {}
    for ev in run.systems:
        for g in ev.get("gpus") or []:
            seen.setdefault(int(g.get("index", 0)), str(g.get("uuid") or "n/a"))
    return [[int(gpu), df["name"].iloc[-1], seen.get(int(gpu), "n/a")]
            for gpu, df in gpu_df.groupby("gpu")]


def _config_part(cfg: dict[str, Any]) -> list[Any]:
    if cfg:
        return [Raw("<details id=\"config\"><summary>Full configuration</summary><pre>"
                    + esc(_config_text(cfg)) + "</pre></details>")]
    return [Raw("<span id=\"config\"></span>"),
            unavailable("The run_start event has no configuration.", "")]


def summary_section(run: RunData, rows: list[dict[str, Any]], gpu_df) -> Raw:
    rs = run.run_start or {}
    env = rs.get("env") or {}
    classes = rs.get("classes") or []
    shown = ", ".join(map(str, classes[:12])) + (", ..." if len(classes) > 12 else "")
    run_pairs = [
        ("Run", run.stamp),
        ("Dataset", rs.get("dataset", "n/a")),
        ("Task", rs.get("task", "n/a")),
        ("Classes", f"{len(classes)}: {shown}" if classes else "n/a"),
        ("Model", rs.get("model", "n/a")),
        ("Strategy", f"{rs.get('mode', 'n/a')} (world size {rs.get('world_size', 'n/a')})"),
        ("Epochs", f"{len(run.epoch_ends)} run of {rs.get('epochs', 'n/a')} planned"),
        ("Final state", _status(run)),
        ("Best (selection metric)", _best(run, rows)),
        ("Duration", _duration_text(run)),
        ("Approx. GPU energy", _energy_text(gpu_df)),
    ]
    devices = env.get("devices") or []
    gpu_rows = _gpu_rows(run, gpu_df)
    env_pairs = [
        ("Host", env.get("host", "n/a")),
        ("Python / torch / timm",
         f"{env.get('python', 'n/a')} / {env.get('torch', 'n/a')} / {env.get('timm', 'n/a')}"),
        ("Git commit", env.get("git_commit") or "n/a (not recorded)"),
        ("Devices (run_start)", ", ".join(map(str, devices)) or "n/a"),
    ]
    parts: list[Any] = [kv_table(run_pairs), Raw("<h3 id=\"env\">Environment</h3>"),
                        kv_table(env_pairs)]
    if gpu_rows:
        parts.append(_scroll(table(["GPU", "Name", "UUID"], gpu_rows)))
    else:
        parts.append(note("GPU UUIDs come from the system samples; this run has none."))
    parts += _config_part(run.config)
    return section("sec-summary", "1. Summary", *parts)


# ── 2. Training ────────────────────────────────────────────────────────────────

def _epoch_table(rows: list[dict[str, Any]]) -> Raw:
    cols = [("train_loss", "train loss"), ("val_loss", "val loss"), ("val_f1", "val F1"),
            ("val_auc", "val AUC"), ("val_accuracy", "val acc."), ("epoch_time_s", "time (s)")]
    cols = [c for c in cols if fg.has(rows, c[0])]
    body = [[r["epoch"], *[num(r.get(k), 4 if k != "epoch_time_s" else 1) for k, _ in cols],
             "best" if r.get("is_best") else ""] for r in rows]
    return _scroll(table(["epoch", *[h for _, h in cols], ""], body,
                         num_cols=range(1, len(cols) + 1)))


def _alerts(run: RunData) -> Raw:
    if not run.alerts:
        return note("No alert events in this run's stream (the run emitted none).")
    rows = []
    for a in run.alerts:
        rows.append([Raw(f"<span class=\"level-{esc(a.get('level', ''))}\">"
                         f"{esc(a.get('level', 'n/a'))}</span>"), a.get("code", ""),
                     a.get("message", ""), a.get("epoch", ""), a.get("batch", ""),
                     num(a.get("value"), 4), num(a.get("threshold"), 4)])
    return _scroll(table(["level", "code", "message", "epoch", "batch", "value", "threshold"],
                         rows, num_cols=(3, 4, 5, 6)))


def training_section(run: RunData, rows: list[dict[str, Any]], gpu_df, pg: _Page) -> Raw:
    if not rows:
        return section("sec-training", "2. Training", unavailable(
            "The stream has no epoch_end event yet.", "Re-run the report once an epoch ends."))
    parts: list[Any] = [Raw("<h3>Loss and metrics</h3>"),
                        _grid2(pg.fig(fg.loss_fig(rows)), pg.fig(fg.metrics_fig(rows))),
                        _epoch_table(rows),
                        Raw("<h3>Throughput and data loading</h3>")]
    if fg.has(rows, "train_samples_per_s") or run.batches:
        figs = [pg.fig(fg.throughput_fig(rows, run.batches))]
        if fg.has(rows, "train_data_wait_frac"):
            figs.append(pg.fig(fg.wait_fig(rows)))
        parts.append(_grid2(*figs))
        if fg.has(rows, "train_data_wait_frac"):
            worst = max(v for v in fg._col(rows, "train_data_wait_frac") if not math.isnan(v))
            parts.append(note(
                f"Highest DataLoader wait: {100 * worst:.1f} % of the training time of an epoch. "
                "Above 30 % the run is limited by data loading, not by the GPU. In DDP the "
                "figures are those of rank 0."))
    else:
        parts.append(unavailable("No samples/s or DataLoader wait in this run.",
                                 "They are written by the instrumented trainer (epoch_end.train "
                                 "and, with the batch monitor, batch events)."))
    parts.append(Raw("<h3>GPU usage</h3>"))
    figs = fg.gpu_figs(gpu_df) if not gpu_df.empty else []
    if figs:
        parts.append(_grid2(*[pg.fig(f) for f in figs]))
    else:
        parts.append(unavailable("No system samples with GPU data.",
                                 "Keep --trace other than off and monitor.system_interval_s > 0; "
                                 "GPU data needs NVML (pynvml and the NVIDIA driver)."))
    parts += [Raw("<h3>Alerts</h3>"), _alerts(run)]
    return section("sec-training", "2. Training", *parts)


# ── 3. Evaluation ──────────────────────────────────────────────────────────────

def _names(row: np.ndarray, classes: list[str], task: str) -> str:
    row = np.asarray(row).ravel()
    if task == "binary":
        return "positive" if row[0] > 0 else "negative"
    if task == "multiclass":
        return classes[int(row.argmax())]
    on = [classes[i] for i in np.nonzero(row)[0]]
    return ", ".join(on) if on else "none"


def _metrics_block(ev: Evaluation) -> list[Any]:
    inp = ev.inputs
    thr = ("argmax (multiclass)" if inp.task == "multiclass"
           else f"{inp.threshold:.3f} ({inp.threshold_source})")
    out: list[Any] = [Raw(
        "<p class=\"params\">"
        f"Epoch {inp.epoch} &middot; {esc(inp.source)} &middot; n = {ev.n} &middot; task "
        f"{esc(inp.task)} &middot; decision threshold {esc(thr)}<br>"
        f"Intervals: percentile bootstrap, {int(round(100 * (1 - ev.alpha)))} % level, "
        f"n_boot = {ev.n_boot}, random seed = {ev.seed}"
        f"{', stratified by class' if inp.task != 'multilabel' else ', whole-sample resampling'}"
        ".</p>")]
    rows = []
    for key, label in METRIC_NAMES.items():
        m = ev.metrics["metrics"][key]
        deg = f" ({m['n_degenerate']} degenerate)" if m["n_degenerate"] else ""
        rows.append([label, num(m["estimate"]), f"({num(m['ci'][0])}, {num(m['ci'][1])})",
                     f"{m['n_boot']}{deg}"])
    out.append(table(["Metric", "Estimate", "95 % CI", "Valid resamples"], rows,
                     num_cols=(1, 2, 3)))
    out.append(note("AUC and AP average the classes that have positives and negatives. F1, "
                    "sensitivity and specificity are macro averages at the decision threshold. "
                    "ECE and Brier are defined in the calibration block. The percentile "
                    "interval is optimistic with few samples or a metric close to its limit."))
    if ev.n < 100:
        out.append(note(f"Only {ev.n} samples: treat every interval as a rough indication.",
                        "missing"))
    return out


def _curves_block(ev: Evaluation, pg: _Page) -> list[Any]:
    inp = ev.inputs
    out: list[Any] = [_grid2(pg.fig(fg.roc_fig(ev.curves, inp.classes)),
                             pg.fig(fg.pr_fig(ev.curves, inp.classes))),
                      note("Legend values are AUC (ROC) and average precision (PR) per class; "
                           "'micro' pools all (sample, class) pairs.")]
    return out


def _perclass_block(ev: Evaluation) -> list[Any]:
    rows = [[r["class"], r["n_pos"], num(r["prevalence"]), num(r["auc"]), num(r["ap"]),
             num(r["sensitivity"]), num(r["specificity"]), num(r["ppv"]), num(r["npv"]),
             num(r["ece"]), num(r["slope"]), num(r["intercept"])] for r in ev.per_class]
    heads = ["Class", "Positives", "Prev.", "AUC", "AP", "Sens.", "Spec.", "PPV", "NPV",
             "ECE", "Cal. slope", "Cal. intercept"]
    return [_scroll(table(heads, rows, num_cols=range(1, 12))),
            note("Sensitivity, specificity, PPV and NPV at the decision threshold (argmax in "
                 "multiclass), one-vs-rest; PPV and NPV at the prevalence of this set. ECE is the "
                 "class-wise one (reliability of that class's probability). Slope and intercept "
                 "come from the logistic recalibration fit of that class (NaN if it does not "
                 "exist).")]


def _confusion_block(ev: Evaluation, pg: _Page) -> list[Any]:
    view = confusion_view(ev)
    out: list[Any] = [pg.fig(fg.confusion_fig(view)), note(view["note"])]
    if ev.inputs.task == "multilabel":
        rows = [[ev.inputs.classes[i], pc["tp"], pc["fp"], pc["fn"], pc["tn"]]
                for i, pc in enumerate(ev.confusion["per_class"])]
        out.append(table(["Class", "TP", "FP", "FN", "TN"], rows, num_cols=(1, 2, 3, 4)))
    return out


def _slope_reading(slope: float, converged: bool) -> str:
    if not converged or math.isnan(slope):
        return "The calibration fit does not exist (single outcome value, constant or separable)."
    if slope < 0.9:
        return ("Slope below 1: predictions are more extreme than the data support "
                "(overconfidence).")
    if slope > 1.1:
        return "Slope above 1: predictions are less extreme than the data support."
    return "Slope close to 1."


def _calibration_block(ev: Evaluation, pg: _Page) -> list[Any]:
    cal, st, ad = ev.calibration, ev.cal_stats, ev.adaptive
    basis = ("top-label (confidence of the predicted class)" if ev.inputs.task == "multiclass"
             else "every (sample, class) pair as one binary prediction")
    rows = [
        ["ECE (equal-width bins)", num(cal["ece"], 4)],
        ["ACE (equal-mass bins)", num(ad["ece"], 4)],
        ["MCE", num(cal["mce"], 4)],
        ["Brier score", num(cal["brier"], 4)],
        ["Class-wise ECE (mean over classes)", num(ev.class_cal["sce"], 4)],
        ["Calibration slope (SE)", f"{num(st['slope'])} ({num(st['slope_se'])})"],
        ["Calibration intercept (SE)", f"{num(st['intercept'])} ({num(st['intercept_se'])})"],
        ["Calibration-in-the-large (slope fixed at 1)", num(st["intercept_citl"])],
    ]
    return [_grid2(pg.fig(fg.reliability_fig(cal)), pg.fig(fg.confidence_hist_fig(cal))),
            table(["Measure", "Value"], rows, num_cols=(1,)),
            note(f"Basis: {basis}; {ev.bins} bins. Slope and intercept: logistic regression of "
                 "the outcome on the logit of the predicted probability (Cox 1958; Van Calster "
                 "et al. 2019); ideal values 1 and 0. " + _slope_reading(st["slope"],
                                                                        bool(st["converged"])))]


def _op_table(op: dict[str, Any], classes: list[str], with_threshold: bool) -> Raw:
    heads = ["Class", *(["Threshold"] if with_threshold else []), "Sens.", "Spec.", "PPV", "NPV"]
    rows = []
    for c, name in enumerate(classes):
        rows.append([name, *([num(op["threshold"][c], 4)] if with_threshold else []),
                     num(op["sensitivity"][c]), num(op["specificity"][c]), num(op["ppv"][c]),
                     num(op["npv"][c])])
    macro = op["macro"]
    rows.append(["macro", *(["" ] if with_threshold else []), num(macro["sensitivity"]),
                 num(macro["specificity"]), num(macro["ppv"]), num(macro["npv"])])
    return _scroll(table(heads, rows, num_cols=range(1, len(heads))))


def _operating_block(ev: Evaluation) -> list[Any]:
    inp, classes = ev.inputs, ev.inputs.classes
    fixed_label = ("Argmax rule (multiclass)" if inp.task == "multiclass"
                   else f"Validation threshold ({inp.threshold:.3f})")
    out: list[Any] = [Raw(f"<h4>{esc(fixed_label)}</h4>"),
                      _op_table(ev.op["fixed"], classes, False)]
    for res in ev.op["sensitivity"]:
        out += [Raw(f"<h4>Highest threshold with sensitivity &ge; {res['target']:.2f}</h4>"),
                _op_table(res, classes, True)]
    out.append(note("Thresholds for a target sensitivity are taken from the ROC points of these "
                    "same data (one-vs-rest, per class), so they are optimistic. For a clinical "
                    "claim, fix the threshold on validation and apply it to a held-out test set "
                    "(paravit eval). PPV and NPV are at the observed prevalence, which is not "
                    "that of the intended population unless the set was sampled from it."))
    return out


def _subgroup_block() -> list[Any]:
    return [unavailable(
        "Performance by subgroup (sex, centre, age band...) needs one attribute per sample, and "
        "the event stream and the inspection files do not record any.",
        "Supply it with the dataset metadata; src.analysis.subgroup_metrics computes metrics, "
        "intervals and the max-min gap from it.")]


def _thumb(mon: an.MonitorSet, i: int, size: int = 128) -> np.ndarray:
    from src.web.inspector_data import upscale
    return upscale(mon.thumbs[i], size)


def _errors_block(ev: Evaluation, k: int) -> list[Any]:
    inp = ev.inputs
    mon, data = inp.monitor, inp.epoch_data
    thr = 0.5 if np.isnan(inp.threshold) else inp.threshold
    worst = an.worst_errors(data.probs, mon.targets, data.loss, inp.task, k, thr)
    out: list[Any] = [note(f"The {len(worst['indices'])} samples of the monitor subset "
                           f"(n = {len(data.probs)}) with the highest loss at epoch {inp.epoch}.")]
    if mon.thumbs is None:
        return out + [unavailable("The monitor set has no thumbnails.", NEED_MONITOR)]
    cards = []
    for j, i in enumerate(worst["indices"]):
        cap = (f"<strong>#{int(mon.indices[i])}</strong> loss {num(worst['loss'][j])}<br>"
               f"true: {esc(_names(worst['targets'][j], inp.classes, inp.task))}<br>"
               f"predicted: {esc(_names(worst['preds'][j], inp.classes, inp.task))}")
        cards.append(f"<figure class=\"card\"><img alt=\"worst error {j + 1}\" "
                     f"src=\"{png_data_uri(_thumb(mon, int(i)))}\"><figcaption>{cap}"
                     "</figcaption></figure>")
    out.append(Raw("<div class=\"grid\">" + "".join(cards) + _DIV_END))
    return out


def _gradcam_block(inp: EvalInputs) -> list[Any]:
    mon, data = inp.monitor, inp.epoch_data
    if data.cam is None or mon.panel is None or mon.thumbs is None or len(mon.panel) == 0:
        return [unavailable("No Grad-CAM panel was saved for this epoch.", CAM_FLAG)]
    from src.web.inspector_data import cam_overlay
    thr = 0.5 if np.isnan(inp.threshold) else inp.threshold
    p, _ = check_pair(data.probs, mon.targets)
    pred = predict(p, inp.task, thr)
    cards = []
    for n, pos in enumerate(mon.panel[:16]):
        pos = int(pos)
        base = _thumb(mon, pos)
        heat = cam_overlay(mon.thumbs[pos], data.cam[n], alpha=0.5, size=128)
        cls = int(data.cam_class[n]) if data.cam_class is not None else -1
        explained = inp.classes[cls] if 0 <= cls < len(inp.classes) else "n/a"
        cap = (f"<strong>#{int(mon.indices[pos])}</strong><br>"
               f"true: {esc(_names(mon.targets[pos], inp.classes, inp.task))}<br>"
               f"predicted: {esc(_names(pred[pos], inp.classes, inp.task))}<br>"
               f"explained class: {esc(explained)}")
        cards.append("<figure class=\"card\"><div class=\"pair\">"
                     f"<img alt=\"input\" src=\"{png_data_uri(base)}\">"
                     f"<img alt=\"Grad-CAM overlay\" src=\"{png_data_uri(heat)}\"></div>"
                     f"<figcaption>{cap}</figcaption></figure>")
    return [Raw("<div class=\"grid\">" + "".join(cards) + _DIV_END),
            note("Input and Grad-CAM overlay of the fixed monitor panel, for the class the "
                 "model predicted. Grad-CAM shows where the model looked, not that it looked at "
                 "the right evidence; it has not been checked here with a sanity test.")]


def evaluation_section(run: RunData, epoch: int | None, n_boot: int, seed: int, bins: int,
                       k_errors: int, pg: _Page) -> tuple[Raw, Evaluation | None]:
    title = "3. Evaluation"
    if run.inspect_dir is None:
        return section("sec-evaluation", title, unavailable(
            "No inspection directory (inspect_<stamp>/) next to the events file, so there are no "
            "per-sample predictions to evaluate.", NEED_MONITOR)), None
    try:
        inp = load_inputs(run.inspect_dir, epoch)
        ev = evaluate(inp, n_boot=n_boot, seed=seed, bins=bins)
    except (NoInspection, FileNotFoundError, ValueError) as exc:
        if epoch is not None and isinstance(exc, NoInspection):
            raise                        # an explicit --epoch that does not exist is a user error
        return section("sec-evaluation", title, unavailable(
            f"The inspection directory cannot be read ({exc}).", NEED_MONITOR)), None
    blocks = [
        ("eval-metrics", "Metrics with confidence intervals", _metrics_block(ev)),
        ("eval-curves", "ROC and precision-recall", _curves_block(ev, pg)),
        ("eval-perclass", "Per-class performance", _perclass_block(ev)),
        ("eval-confusion", "Confusion matrix", _confusion_block(ev, pg)),
        ("eval-calibration", "Calibration", _calibration_block(ev, pg)),
        ("eval-operating", "Operating points", _operating_block(ev)),
        ("eval-subgroups", "Subgroups", _subgroup_block()),
        ("eval-errors", "Worst errors", _errors_block(ev, k_errors)),
        ("eval-gradcam", "Grad-CAM panel", _gradcam_block(inp)),
    ]
    parts = [Raw(f"<div id=\"{i}\"><h3>{esc(t)}</h3>{''.join(esc(p) for p in body)}</div>")
             for i, t, body in blocks]
    return section("sec-evaluation", title, *parts), ev


# ── 4. Dynamics ────────────────────────────────────────────────────────────────

def _cartography_parts(run: RunData, epochs: list[int], pg: _Page) -> list[Any]:
    if len(epochs) < 2:
        return [unavailable(
            f"Cartography needs at least two inspected epochs; this run has {len(epochs)}.",
            "Lower monitor.every or train for more epochs.")]
    pt, ok = dynamics_inputs(run.inspect_dir, epochs)
    carto = an.cartography(pt, ok)
    counts = {r: int((carto["region"] == r).sum()) for r in ("easy", "ambiguous", "hard")}
    parts: list[Any] = [
        pg.fig(fg.cartography_fig(carto)),
        table(["Region", "Samples", "Share"],
              [[r, n, f"{100 * n / max(len(carto['region']), 1):.1f} %"]
               for r, n in counts.items()], num_cols=(1, 2)),
        note(f"{len(carto['region'])} monitor samples over epochs {epochs[0]} to "
             f"{epochs[-1]} ({len(epochs)} inspected). Ambiguous: variability above "
             f"{num(carto['thresholds']['variability'], 3)}; of the rest, easy if "
             f"confidence is at least {carto['thresholds']['confidence']:.2f}. "
             "Consistently 'hard' samples are candidates for label noise.")]
    mon = an.load_monitor_set(run.inspect_dir)
    a, b = an.load_epoch(run.inspect_dir, epochs[-2]), an.load_epoch(run.inspect_dir, epochs[-1])
    thr = b.threshold if b.threshold is not None else 0.5
    fl = an.flips(a.probs, b.probs, mon.targets, mon.task, thr)
    parts.append(note(f"Between epochs {epochs[-2]} and {epochs[-1]}: "
                      f"{len(fl['to_wrong'])} samples went from right to wrong and "
                      f"{len(fl['to_right'])} from wrong to right."))
    return parts


def _embedding_parts(ev: Evaluation | None, pg: _Page) -> list[Any]:
    if ev is None or ev.inputs.epoch_data.embeddings is None:
        return [unavailable("No embeddings were saved.",
                            "Set monitor.embeddings: true (the default) with --monitor.")]
    inp = ev.inputs
    mon, data = inp.monitor, inp.epoch_data
    xy = an.project_2d(data.embeddings, "pca")
    thr = 0.5 if np.isnan(inp.threshold) else inp.threshold
    if inp.task == "multilabel":
        right = an.sample_correct(data.probs, mon.targets, inp.task, thr)
        labels = ["exact match" if r else "not exact" for r in right]
        what = "coloured by whether all labels are right"
    else:
        labels = [_names(mon.targets[i], inp.classes, inp.task) for i in range(len(xy))]
        what = "coloured by true class"
    return [pg.fig(fg.embedding_fig(xy, labels, f"PCA of the embeddings, epoch {inp.epoch}")),
            note(f"Input of the last linear layer for the monitor subset, projected on its "
                 f"first two principal components, {what}.")]


def dynamics_section(run: RunData, ev: Evaluation | None, pg: _Page) -> Raw:
    title = "4. Training dynamics"
    if run.inspect_dir is None:
        return section("sec-dynamics", title, unavailable(
            "No inspection directory, so no per-epoch predictions or embeddings.", NEED_MONITOR))
    epochs = an.list_epochs(run.inspect_dir)
    parts: list[Any] = [Raw("<h3 id=\"dyn-carto\">Data cartography</h3>"),
                        *_cartography_parts(run, epochs, pg),
                        Raw("<h3 id=\"dyn-embed\">Embeddings</h3>"), *_embedding_parts(ev, pg)]
    return section("sec-dynamics", title, *parts)


# ── 5. TRIPOD+AI ───────────────────────────────────────────────────────────────

_ANCHOR_NAMES = {"sec-summary": "Summary", "env": "Summary, environment", "config": "Summary, "
                 "configuration", "sec-training": "Training", "sec-evaluation": "Evaluation",
                 "eval-metrics": "Evaluation, metrics", "eval-perclass": "Evaluation, per class",
                 "eval-operating": "Evaluation, operating points"}


def tripod_section(run: RunData, ev: Evaluation | None) -> Raw:
    env = (run.run_start or {}).get("env") or {}
    ctx = tripod.Context(has_start=run.run_start is not None, has_config=bool(run.config),
                         has_eval=ev is not None, commit=env.get("git_commit"),
                         n_boot=ev.n_boot if ev else 0, has_system=bool(run.systems))
    items = tripod.assess(ctx)
    count = tripod.summary(items)
    rows = []
    for it in items:
        where = (Raw(f"<a href=\"#{esc(it.anchor)}\">{esc(_ANCHOR_NAMES.get(it.anchor, it.anchor))}"
                     "</a>") if it.anchor else "")
        rows.append([it.id, Raw(f"<strong>{esc(it.section)}.</strong> {esc(it.text)}"), it.applies,
                     Raw(f"<span class=\"badge {it.status}\">"
                         f"{esc(tripod.STATUS_LABEL[it.status])}</span>"), where, it.note])
    intro = Raw(
        f"<p>Checklist of {esc(tripod.REFERENCE)} "
        f"<a href=\"{tripod.DOI_URL}\">doi:10.1136/bmj-2023-078378</a>. The run can document "
        "part of it by itself; the rest needs the authors. An item is marked <em>documented</em> "
        "only when the report covers everything it asks for apart from a free-text rationale; "
        "<em>partly documented</em> says what is covered and what is not; <em>pending</em> "
        "means nothing in the logs can answer it. This is an aid to fill the checklist, not a "
        "statement of compliance.</p>")
    summ = note(f"{len(items)} items: {count['documented']} documented, {count['partial']} partly "
                f"documented, {count['pending']} pending. Applies: D = development, E = "
                "evaluation, D;E = both.")
    return section("sec-tripod", "5. TRIPOD+AI checklist", intro, summ, _scroll(table(
        ["Item", "Requirement (abridged)", "Applies", "Status", "Where", "Notes"], rows)))


# ── Entry points ───────────────────────────────────────────────────────────────

def build_report(run: RunData, *, epoch: int | None = None, n_boot: int = 1000, seed: int = 0,
                 bins: int = 15, k_errors: int = 8, now: _dt.datetime | None = None) -> str:
    """The complete HTML document of a run.

    ``epoch``: inspected epoch to evaluate (default the last); an epoch that was not inspected
    raises :class:`NoInspection`. The bootstrap uses ``seed`` and says so in the report.
    """
    import plotly.offline as po

    now = now or _dt.datetime.now(_dt.timezone.utc)
    pg = _Page()
    rows = epoch_rows(run)
    gpu_df, _ = _system_frames(run)
    summary = summary_section(run, rows, gpu_df)
    training = training_section(run, rows, gpu_df, pg)
    evaluation, ev = evaluation_section(run, epoch, n_boot, seed, bins, k_errors, pg)
    dynamics = dynamics_section(run, ev, pg)
    check = tripod_section(run, ev)
    rs = run.run_start or {}
    title = f"Run report: {rs.get('model', 'unknown model')} on {rs.get('dataset', 'unknown dataset')}"
    sub = (f"Run {run.stamp} | {rs.get('task', 'task n/a')} | source: {run.events_path.name}"
           + (f" + {run.inspect_dir.name}" if run.inspect_dir else ""))
    footer = (f"Generated by paravit report (format v{REPORT_VERSION}) on "
              f"{now.strftime('%Y-%m-%d %H:%M UTC')} from the run's logs only; nothing was "
              "trained or measured for this document.")
    body = "".join(str(s) for s in (summary, training, evaluation, dynamics, check))
    return page(title, sub, SECTIONS, body, po.get_plotlyjs(), footer)


def default_output(events_path: Path, stamp: str) -> Path:
    return events_path.parent / f"report_{stamp}.html"


def write_report(spec: str | Path, output: str | Path | None = None, root: str | Path = ".",
                 **options: Any) -> Path:
    """Resolve ``spec`` (events file, run folder or stamp), build the report and write it."""
    events = resolve_run(spec, root)
    run = load_run(events)
    html_text = build_report(run, **options)
    out = Path(output) if output else default_output(events, run.stamp)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return out
