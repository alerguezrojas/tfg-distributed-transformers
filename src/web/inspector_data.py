"""Pure helpers behind the "Inspector" page (no Streamlit, no torch: unit-testable).

Three kinds of input, all produced by the trainer and only read here (docs/inspeccion.md):
the folded events of a run (:class:`~src.web.events_reader.RunState`: ``system``, ``batch``,
``layers``, ``histograms``…), the ``inspect_<ts>/`` arrays loaded with ``src.analysis.io`` and the
per-sample analysis of ``src.analysis``. This module turns them into DataFrames, matrices and
images ready to plot; the page modules only draw.
"""
from __future__ import annotations

import math

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.web.events_reader import DEFAULT_STALLED_AFTER_S, discover_event_files
from src.web.run_registry import (
    _TIMESTAMP_RE, RunInfo, _env_mode_model_from_path, timestamp_sort_key,
)

#: Fraction of the training time spent waiting for the DataLoader above which the run is
#: considered I/O-bound (docs/inspeccion.md: ">0,3 indica cuello de I/O").
DATA_WAIT_IO_BOUND = 0.3
#: Below this the input pipeline is clearly not a bottleneck (compute-bound).
DATA_WAIT_COMPUTE_BOUND = 0.1
#: Healthy band of the update ratio ‖Δw‖/‖w‖ (a common rule of thumb for SGD/Adam).
HEALTHY_UPDATE_RATIO = (1e-4, 1e-2)

GREEN, RED, AMBER = (46, 139, 87), (176, 65, 62), (197, 123, 39)


# ── Discovery ───────────────────────────────────────────────────────────────────

@dataclass
class InspectorRun:
    """A run the Inspector can open: an events stream, an inspection directory, or both."""

    timestamp: str
    label: str
    events_path: Path | None = None
    inspect_dir: Path | None = None
    active: bool = False


def _label(ts: str, path: Path, logs_root: Path) -> str:
    env, mode, model, dataset = _env_mode_model_from_path(path, logs_root)
    return RunInfo(timestamp=ts, log_path=path, trace_mode="simple", env=env, mode=mode,
                   model=model, dataset=dataset).label


def discover_inspector_runs(root: Path, now: float | None = None,
                            stalled_after_s: float = DEFAULT_STALLED_AFTER_S
                            ) -> list[InspectorRun]:
    """Runs under ``<root>/logs`` that have an events file or an ``inspect_*`` directory.

    Active runs first, then newest first. An events file and the ``inspect_<ts>/`` directory
    next to it (same timestamp) are one run.
    """
    logs_root = root / "logs"
    out: dict[tuple[str, Path], InspectorRun] = {}
    for ef in discover_event_files(root, now=now, stalled_after_s=stalled_after_s):
        inspect = ef.path.parent / f"inspect_{ef.timestamp}"
        out[(ef.timestamp, ef.path.parent)] = InspectorRun(
            timestamp=ef.timestamp, label=ef.label, events_path=ef.path,
            inspect_dir=inspect if inspect.is_dir() else None, active=ef.active)
    if logs_root.exists():
        for d in logs_root.rglob("inspect_*"):
            m = _TIMESTAMP_RE.search(d.name)
            if not d.is_dir() or not m or (m.group(1), d.parent) in out:
                continue
            out[(m.group(1), d.parent)] = InspectorRun(
                timestamp=m.group(1), label=_label(m.group(1), d / "monitor_set.npz", logs_root),
                inspect_dir=d)
    runs = sorted(out.values(), key=lambda r: timestamp_sort_key(r.timestamp), reverse=True)
    runs.sort(key=lambda r: not r.active)       # stable: active first
    return runs


def resolve_inspect_dir(events_path: Path | None, monitor_start: dict[str, Any] | None,
                        root: Path | None = None) -> Path | None:
    """Directory announced by ``monitor_start.dir`` (relative to the log), if it exists."""
    if events_path is None or not monitor_start or not monitor_start.get("dir"):
        return None
    rel = Path(str(monitor_start["dir"]))
    candidates = [rel] if rel.is_absolute() else [events_path.parent / rel]
    if root is not None and not rel.is_absolute():
        candidates.append(root / rel)
    return next((c for c in candidates if c.is_dir()), None)


# ── System ──────────────────────────────────────────────────────────────────────

GPU_FIELDS = ("util_pct", "mem_used_mb", "mem_total_mb", "power_w", "temp_c", "sm_clock_mhz")
HOST_FIELDS = ("cpu_pct", "ram_used_pct", "proc_cpu_pct", "proc_rss_mb")


def _num(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")


def system_frames(events: Iterable[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-GPU samples, host samples) from ``system`` events.

    GPU frame: ``t_s`` (seconds since the first sample), ``gpu`` (index), ``name``, the
    :data:`GPU_FIELDS` and ``mem_pct``. Host frame: ``t_s`` + :data:`HOST_FIELDS`. Events
    without GPUs (no NVML) only contribute to the host frame.
    """
    gpu_rows, host_rows = [], []
    t0 = None
    for ev in events:
        t = _num(ev.get("time"))
        if math.isnan(t):
            continue
        t0 = t if t0 is None else t0
        host_rows.append({"t_s": t - t0, **{k: _num(ev.get(k)) for k in HOST_FIELDS}})
        for g in ev.get("gpus") or []:
            row = {"t_s": t - t0, "gpu": int(_num(g.get("index"))) if g.get("index") is not None
                   else 0, "name": str(g.get("name", "")),
                   **{k: _num(g.get(k)) for k in GPU_FIELDS}}
            total = row["mem_total_mb"]
            row["mem_pct"] = 100.0 * row["mem_used_mb"] / total if total and not math.isnan(total) \
                else float("nan")
            gpu_rows.append(row)
    return (pd.DataFrame(gpu_rows, columns=["t_s", "gpu", "name", *GPU_FIELDS, "mem_pct"]),
            pd.DataFrame(host_rows, columns=["t_s", *HOST_FIELDS]))


def energy_wh(gpu_df: pd.DataFrame) -> dict[int, float]:
    """Approximate energy per GPU (Wh): trapezoidal integral of the sampled power over time.

    An estimate bounded by the sampling interval (a few seconds); it is *not* the metered
    energy of ``measure_energy``. GPUs with fewer than two samples are omitted.
    """
    out: dict[int, float] = {}
    if gpu_df.empty:
        return out
    for gpu, df in gpu_df.groupby("gpu"):
        df = df.dropna(subset=["power_w"]).sort_values("t_s")
        if len(df) < 2:
            continue
        t, p = df["t_s"].to_numpy(), df["power_w"].to_numpy()
        joules = float(np.sum(np.diff(t) * (p[1:] + p[:-1]) / 2))
        out[int(gpu)] = joules / 3600.0
    return out


def system_summary(gpu_df: pd.DataFrame) -> pd.DataFrame:
    """One row per GPU: mean and peak of utilisation, memory, power and temperature."""
    if gpu_df.empty:
        return pd.DataFrame()
    rows = []
    for gpu, df in gpu_df.groupby("gpu"):
        rows.append({
            "GPU": int(gpu), "Name": df["name"].iloc[-1],
            "Util mean (%)": df["util_pct"].mean(), "Util peak (%)": df["util_pct"].max(),
            "Memory mean (MB)": df["mem_used_mb"].mean(),
            "Memory peak (MB)": df["mem_used_mb"].max(),
            "Power mean (W)": df["power_w"].mean(), "Power peak (W)": df["power_w"].max(),
            "Temp peak (°C)": df["temp_c"].max(),
        })
    return pd.DataFrame(rows)


# ── Performance ─────────────────────────────────────────────────────────────────

def batch_performance(batch_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Per-batch throughput and time split from ``RunState.batch_rows()``.

    Keeps the rows that carry the new fields; adds ``data_frac`` = data / (data + step).
    Empty if the trainer predates them.
    """
    df = pd.DataFrame(batch_rows)
    cols = [c for c in ("samples_per_s", "data_time_s", "step_time_s") if c in df.columns]
    if df.empty or not cols or "progress" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=cols, how="all")
    if {"data_time_s", "step_time_s"} <= set(df.columns):
        total = df["data_time_s"] + df["step_time_s"]
        df = df.assign(data_frac=(df["data_time_s"] / total).where(total > 0))
    return df.reset_index(drop=True)


def epoch_performance(epoch_ends: list[dict[str, Any]]) -> pd.DataFrame:
    """``epoch``, ``samples_per_s`` and ``data_wait_frac`` of each finished epoch (if present)."""
    rows = []
    for ev in epoch_ends:
        tr = ev.get("train") or {}
        rows.append({"epoch": ev.get("epoch"), "samples_per_s": _num(tr.get("samples_per_s")),
                     "data_wait_frac": _num(tr.get("data_wait_frac")),
                     "epoch_time_s": _num(ev.get("epoch_time_s"))})
    df = pd.DataFrame(rows, columns=["epoch", "samples_per_s", "data_wait_frac", "epoch_time_s"])
    return df.dropna(subset=["samples_per_s", "data_wait_frac"], how="all").reset_index(drop=True)


def diagnose_data_wait(frac: float | None) -> tuple[str, str]:
    """(level, text) for the fraction of training time spent waiting for data.

    ``level`` is ``bad`` (I/O-bound, > :data:`DATA_WAIT_IO_BOUND`), ``ok`` (compute-bound,
    < :data:`DATA_WAIT_COMPUTE_BOUND`), ``warn`` in between, or ``none`` without data.
    """
    if frac is None or math.isnan(frac):
        return "none", "No data-wait measurement (needs a trainer that logs data_wait_frac)."
    pct = f"{100 * frac:.0f}%"
    if frac > DATA_WAIT_IO_BOUND:
        return "bad", (f"I/O-bound: {pct} of the training time is spent waiting for the "
                       f"DataLoader (> {100 * DATA_WAIT_IO_BOUND:.0f}%). More workers, a faster "
                       "disk or caching the dataset will help more than a faster GPU.")
    if frac < DATA_WAIT_COMPUTE_BOUND:
        return "ok", (f"Compute-bound: only {pct} of the time is spent waiting for data. "
                      "The GPU is the limit, as it should be.")
    return "warn", (f"Mostly compute-bound, but {pct} of the time goes to waiting for data: "
                    "close to the I/O threshold "
                    f"({100 * DATA_WAIT_IO_BOUND:.0f}%).")


# ── Layers ──────────────────────────────────────────────────────────────────────

LAYER_METRICS = {"grad_norm": "Gradient norm", "weight_norm": "Weight norm",
                 "update_ratio": "Update ratio"}


def layer_matrix(layer_events: Iterable[dict[str, Any]], metric: str,
                 batches_per_epoch: dict[int, int] | None = None, max_cols: int = 300
                 ) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Group × time matrix of one layer statistic.

    Returns ``(x, groups, Z)``: ``x`` = fractional epoch of each ``layers`` event
    (``epoch - 1 + batch / n_batches``; ``n_batches`` from ``batches_per_epoch`` or, failing
    that, the largest batch index seen in that epoch), ``groups`` in order of first appearance
    and ``Z`` of shape (groups, len(x)) with NaN where a group is missing. At most ``max_cols``
    columns: longer series are thinned evenly.
    """
    events = [e for e in layer_events if isinstance(e.get("groups"), list)]
    if not events:
        return np.empty(0), [], np.empty((0, 0))
    seen_max: dict[int, int] = {}
    for e in events:
        ep, b = e.get("epoch"), e.get("batch")
        if isinstance(ep, int) and isinstance(b, (int, float)):
            seen_max[ep] = max(seen_max.get(ep, 0), int(b))
    groups: list[str] = []
    for e in events:
        for g in e["groups"]:
            if g.get("name") not in groups and g.get("name") is not None:
                groups.append(g["name"])
    index = {g: i for i, g in enumerate(groups)}
    xs, cols = [], []
    for e in events:
        ep, b = e.get("epoch"), e.get("batch")
        n = (batches_per_epoch or {}).get(ep) or seen_max.get(ep) or 1
        xs.append((ep - 1) + min(_num(b) / n, 1.0) if isinstance(ep, int) else float("nan"))
        col = np.full(len(groups), np.nan)
        for g in e["groups"]:
            if g.get("name") in index:
                col[index[g["name"]]] = _num(g.get(metric))
        cols.append(col)
    x, z = np.array(xs), np.stack(cols, axis=1)
    if x.size > max_cols:
        keep = np.unique(np.linspace(0, x.size - 1, max_cols).round().astype(int))
        x, z = x[keep], z[:, keep]
    return x, groups, z


def update_ratio_status(value: float | None) -> str:
    """``low`` / ``healthy`` / ``high`` against :data:`HEALTHY_UPDATE_RATIO`, or ``unknown``."""
    if value is None or math.isnan(value) or value <= 0:
        return "unknown"
    lo, hi = HEALTHY_UPDATE_RATIO
    return "low" if value < lo else "high" if value > hi else "healthy"


def histograms_for_epoch(histograms: dict[int, dict[str, Any]], epoch: int
                         ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{group: (edges, counts)}`` of one epoch's weight histograms (empty if none)."""
    ev = histograms.get(epoch)
    out = {}
    for g in (ev or {}).get("groups") or []:
        edges, counts = np.asarray(g.get("edges", []), float), np.asarray(g.get("counts", []), float)
        if g.get("name") and edges.size == counts.size + 1 and counts.size:
            out[g["name"]] = (edges, counts)
    return out


# ── Sample tables ───────────────────────────────────────────────────────────────

def _names(classes: list[str], idx: Iterable[int]) -> str:
    return ", ".join(classes[i] for i in idx) or "none"


def prediction_table(probs, targets, loss, task: str, classes: list[str], threshold: float = 0.5,
                     prev_probs=None) -> pd.DataFrame:
    """One row per monitored sample for the gallery.

    Columns: ``pos`` (position in the monitor set), ``true`` / ``pred`` (class names; several
    joined by commas in multi-label), ``true_idx`` / ``pred_idx`` (tuples of class indices),
    ``confidence``, ``correct``, ``loss`` and ``flip`` (``to_wrong`` / ``to_right`` / ``""``
    against ``prev_probs``). Confidence: top probability (multiclass, multi-label) or the
    probability of the predicted side (binary). Correct = exact match of every label (see
    ``src.analysis.dynamics.sample_correct``).
    """
    from src.analysis._common import check_pair, predict
    from src.analysis.dynamics import flips, sample_correct

    p, y = check_pair(probs, targets)
    pred = predict(p, task, threshold)
    if task == "binary":
        conf = np.maximum(p[:, 0], 1 - p[:, 0])
    else:
        conf = p.max(axis=1)
    flip = np.full(len(p), "", dtype=object)
    if prev_probs is not None:
        f = flips(prev_probs, probs, targets, task, threshold)
        flip[f["to_wrong"]] = "to_wrong"
        flip[f["to_right"]] = "to_right"
    true_idx = [tuple(np.nonzero(r)[0]) for r in y]
    pred_idx = [tuple(np.nonzero(r)[0]) for r in pred]
    if task == "binary":
        true_s = ["positive" if t else "negative" for t in y[:, 0]]
        pred_s = ["positive" if t else "negative" for t in pred[:, 0]]
    else:
        true_s = [_names(classes, t) for t in true_idx]
        pred_s = [_names(classes, t) for t in pred_idx]
    return pd.DataFrame({
        "pos": np.arange(len(p)), "true": true_s, "pred": pred_s, "true_idx": true_idx,
        "pred_idx": pred_idx, "confidence": conf,
        "correct": sample_correct(probs, targets, task, threshold),
        "loss": np.asarray(loss, dtype=float), "flip": flip,
    })


SORTS = {"Position": ("pos", True), "Loss (highest first)": ("loss", False),
         "Loss (lowest first)": ("loss", True), "Confidence (highest first)": ("confidence", False),
         "Confidence (lowest first)": ("confidence", True)}


def filter_table(df: pd.DataFrame, true_class: int | None = None, pred_class: int | None = None,
                 outcome: str = "all", flips_only: str = "all", sort: str = "Position"
                 ) -> pd.DataFrame:
    """Applies the gallery filters. ``outcome``: all | correct | wrong. ``flips_only``: all |
    changed | to_wrong | to_right. ``true_class`` / ``pred_class``: class index or None."""
    mask = pd.Series(True, index=df.index)
    if true_class is not None:
        mask &= df["true_idx"].map(lambda t: true_class in t)
    if pred_class is not None:
        mask &= df["pred_idx"].map(lambda t: pred_class in t)
    if outcome == "correct":
        mask &= df["correct"]
    elif outcome == "wrong":
        mask &= ~df["correct"]
    if flips_only == "changed":
        mask &= df["flip"] != ""
    elif flips_only in ("to_wrong", "to_right"):
        mask &= df["flip"] == flips_only
    col, asc = SORTS.get(sort, SORTS["Position"])
    return df[mask].sort_values(col, ascending=asc, kind="stable")


def trajectory_frame(epochs: list[int], probs_by_epoch, pos: int, classes: list[str]
                     ) -> pd.DataFrame:
    """Long frame ``epoch, class, class_idx, prob`` of one sample across epochs."""
    rows = []
    for ep, p in zip(epochs, probs_by_epoch):
        row = np.asarray(p, dtype=float)[pos]
        rows += [{"epoch": ep, "class": classes[c] if c < len(classes) else str(c),
                  "class_idx": c, "prob": float(row[c])} for c in range(len(row))]
    return pd.DataFrame(rows, columns=["epoch", "class", "class_idx", "prob"])


def primary_label(targets, classes: list[str]) -> np.ndarray:
    """One label per sample for colouring: the class, ``multiple`` or ``none``."""
    y = np.asarray(targets) > 0
    out = np.full(len(y), "none", dtype=object)
    n = y.sum(axis=1)
    out[n == 1] = [classes[i] for i in y[n == 1].argmax(axis=1)]
    out[n > 1] = "multiple"
    return out


def predicted_label(probs, classes: list[str]) -> np.ndarray:
    """Top-1 class name per sample."""
    return np.array([classes[i] for i in np.asarray(probs, float).argmax(axis=1)], dtype=object)


# ── Images ──────────────────────────────────────────────────────────────────────

def upscale(img: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of an (h, w, 3) uint8 image to (size, size, 3)."""
    h, w = img.shape[:2]
    yi = (np.arange(size) * h // size).clip(0, h - 1)
    xi = (np.arange(size) * w // size).clip(0, w - 1)
    return img[yi][:, xi]


def add_border(img: np.ndarray, color: tuple[int, int, int], width: int = 4) -> np.ndarray:
    """Copy of ``img`` with a coloured frame drawn over its outer ``width`` pixels."""
    out = img.copy()
    out[:width, :], out[-width:, :] = color, color
    out[:, :width], out[:, -width:] = color, color
    return out


def outcome_card(thumb: np.ndarray, correct: bool, size: int = 128) -> np.ndarray:
    """Gallery card: enlarged thumbnail with a green (right) or red (wrong) frame."""
    return add_border(upscale(thumb, size), GREEN if correct else RED)


def _jet(x: np.ndarray) -> np.ndarray:
    """(…,) values in [0, 1] → (…, 3) uint8 'jet'-like colours."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def cam_overlay(thumb: np.ndarray, cam: np.ndarray, alpha: float = 0.5, size: int | None = None
                ) -> np.ndarray:
    """Grad-CAM heatmap blended over the thumbnail.

    ``cam`` is (h, w) uint8 (0-255, any grid: 14×14 ViT patches, a CNN's last map…); it is
    bilinearly resized to the image (``size`` px, default the thumbnail's side) and blended
    with weight ``alpha`` (0 = image only, 1 = heatmap only).
    """
    from PIL import Image

    side = size or thumb.shape[0]
    base = upscale(thumb, side) if thumb.shape[0] != side else thumb
    heat = np.asarray(Image.fromarray(np.asarray(cam, dtype=np.uint8)).resize(
        (side, side), Image.BILINEAR), dtype=np.float64) / 255.0
    blended = (1 - alpha) * base.astype(np.float64) + alpha * _jet(heat).astype(np.float64)
    return blended.round().astype(np.uint8)


# ── Misc ────────────────────────────────────────────────────────────────────────

def thin_epochs(epochs: list[int], max_n: int) -> list[int]:
    """At most ``max_n`` epochs spread evenly, always keeping the first and the last."""
    if len(epochs) <= max_n or max_n < 2:
        return list(epochs)
    keep = np.unique(np.linspace(0, len(epochs) - 1, max_n).round().astype(int))
    return [epochs[i] for i in keep]


# ── Alerts ──────────────────────────────────────────────────────────────────────

CRITICAL, WARNING = "critical", "warning"
ALERT_COLUMNS = ["time", "level", "code", "group", "epoch", "batch", "value", "threshold", "message"]


def _alert_level(ev: dict[str, Any]) -> str:
    return CRITICAL if ev.get("level") == CRITICAL else WARNING     # unknown level: warning


def _group(ev: dict[str, Any]) -> str:
    """Layer name, ``gpu<N>``, ``train`` / ``val`` or empty (the alert has no group)."""
    g = ev.get("group")
    return "" if g is None else str(g)


def alert_frame(alerts: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Chronological table of ``alert`` events (contract in docs/eventos.md).

    Columns: ``time`` (datetime, UTC), ``level`` (``critical`` / ``warning``; anything else is
    shown as a warning), ``code``, ``group``, ``epoch``, ``batch``, ``value``, ``threshold``, ``message``.
    Missing numeric fields are NaN.
    """
    rows = []
    for ev in alerts:
        rows.append({"time": _num(ev.get("time")), "level": _alert_level(ev),
                     "code": str(ev.get("code", "")), "group": _group(ev), "epoch": _num(ev.get("epoch")),
                     "batch": _num(ev.get("batch")), "value": _num(ev.get("value")),
                     "threshold": _num(ev.get("threshold")), "message": str(ev.get("message", ""))})
    df = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def _time_key(ev: dict[str, Any]) -> float:
    t = _num(ev.get("time"))
    return 0.0 if math.isnan(t) else t


def active_alerts(alerts: Iterable[dict[str, Any]], current_epoch: int | None = None,
                  window: int = 1) -> list[dict[str, Any]]:
    """Alerts to show in the banner: the latest event of each (``code``, ``group``).

    The contract has no "resolved" event, so *active* is defined by recency: an alert stays
    while its epoch is within ``window`` epochs of ``current_epoch`` (or has no epoch). With
    ``current_epoch=None`` (finished run) every (code, group) that fired is kept. Critical first, then the
    most recent. Each returned dict gains a normalised ``level``.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for ev in alerts:                       # the trainer fires one alert per (code, group)
        latest[(str(ev.get("code", "")), _group(ev))] = ev
    out = []
    for ev in latest.values():
        ep = ev.get("epoch")
        if (current_epoch is not None and isinstance(ep, (int, float)) and not isinstance(ep, bool)
                and ep < current_epoch - window):
            continue
        out.append({**ev, "level": _alert_level(ev)})
    out.sort(key=lambda e: (e["level"] != CRITICAL, -_time_key(e)))
    return out


def alert_line(ev: dict[str, Any]) -> str:
    """One banner line: ``epoch 3 · loss_divergence · message``."""
    parts = []
    ep = ev.get("epoch")
    if isinstance(ep, (int, float)) and not isinstance(ep, bool):
        parts.append(f"epoch {int(ep)}")
    parts.append(str(ev.get("code", "alert")) + (f" [{_group(ev)}]" if _group(ev) else ""))
    if ev.get("message"):
        parts.append(str(ev["message"]))
    return " · ".join(parts)


def events_have(state: Any) -> dict[str, bool]:
    """Which event families the run carries (to explain what is missing)."""
    return {
        "system": bool(state.systems), "layers": bool(state.layers),
        "histograms": bool(state.histograms or state.grad_histograms),
        "monitor": state.monitor_start is not None,
    }

