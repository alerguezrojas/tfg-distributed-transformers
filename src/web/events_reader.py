"""Reader for the live-events stream (``events_<DDMMYYYY_HHMMSS>.jsonl``).

While a run trains, rank 0 appends one JSON object per line next to the run's
``train_*.log`` (same timestamp). This module follows that file incrementally and
folds the events into a :class:`RunState` that the "Live" tab renders. Pure: no
Streamlit, no torch, no GPU — the web only reads.

Schema v1 (every line carries ``v``, ``type`` and ``time``): ``run_start``,
``epoch_start``, ``batch``, ``epoch_end``, ``early_stop``, ``heartbeat``, ``run_end``
(docs/eventos.md) and the inspection events ``system``, ``monitor_start``, ``inspection``,
``layers`` and ``histograms`` (docs/inspeccion.md), which the "Inspector" page reads. Unknown
types, and lines from a newer schema version, are ignored so an older dashboard
keeps working with a newer trainer.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.web.run_registry import (
    _DEFAULT_DATASET, _TIMESTAMP_RE, RunInfo, _env_mode_model_from_path, timestamp_sort_key,
)

SUPPORTED_VERSION = 1
MAX_BATCHES = 2000                  # per-batch events kept in memory (most recent)
DEFAULT_STALLED_AFTER_S = 180.0     # no run_end and no writes for this long → stalled
                                    # (a live trainer writes a heartbeat every 30 s)
MAX_SYSTEM = 5000                   # ``system`` samples kept (every few seconds → hours)
MAX_LAYERS = 4000                   # ``layers`` events kept (most recent)
MAX_HISTOGRAMS = 200                # epochs with weight histograms kept (most recent)
MAX_INSPECTIONS = 1000
_TAIL_BYTES = 8192                  # how much of the file end is scanned for run_end

# Derived run states.
RUNNING, COMPLETED, EARLY_STOPPED, FAILED, INTERRUPTED, STALLED, WAITING = (
    "running", "completed", "early_stopped", "failed", "interrupted", "stalled", "waiting")
FINAL_STATES = (COMPLETED, EARLY_STOPPED, FAILED, INTERRUPTED)


def parse_event(line: bytes | str) -> dict[str, Any] | None:
    """One JSONL line → event dict, or None if it is not a usable v1 event."""
    try:
        ev = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(ev, dict) or not isinstance(ev.get("type"), str):
        return None
    v = ev.get("v", SUPPORTED_VERSION)
    if not isinstance(v, int) or isinstance(v, bool) or v > SUPPORTED_VERSION:
        return None
    return ev


@dataclass
class RunState:
    """Everything the Live tab needs, built by folding events in order."""

    run_start: dict[str, Any] | None = None
    epoch_ends: list[dict[str, Any]] = field(default_factory=list)
    batches: deque = field(default_factory=lambda: deque(maxlen=MAX_BATCHES))
    current_epoch: int | None = None        # from the latest epoch_start
    planned_epochs: int | None = None
    early_stop: dict[str, Any] | None = None
    # Inspection events (docs/inspeccion.md): bounded, most recent kept.
    systems: deque = field(default_factory=lambda: deque(maxlen=MAX_SYSTEM))
    layers: deque = field(default_factory=lambda: deque(maxlen=MAX_LAYERS))
    histograms: dict[int, dict[str, Any]] = field(default_factory=dict)   # epoch → event
    inspections: deque = field(default_factory=lambda: deque(maxlen=MAX_INSPECTIONS))
    monitor_start: dict[str, Any] | None = None
    run_end: dict[str, Any] | None = None
    last_event_time: float | None = None    # "time" field of the latest event
    mtime: float | None = None              # file mtime, set by the follower
    n_events: int = 0
    n_ignored: int = 0                      # unparseable lines + unknown types/versions

    # ── folding ────────────────────────────────────────────────────────────────

    def apply(self, ev: dict[str, Any]) -> None:
        """Folds one event (already validated by :func:`parse_event`)."""
        kind = ev["type"]
        if kind == "run_start":
            self.run_start = ev
            self.planned_epochs = _as_int(ev.get("epochs"), self.planned_epochs)
        elif kind == "epoch_start":
            self.current_epoch = _as_int(ev.get("epoch"), self.current_epoch)
            self.planned_epochs = _as_int(ev.get("epochs"), self.planned_epochs)
        elif kind == "batch":
            self.batches.append(ev)
        elif kind == "epoch_end":
            self.planned_epochs = _as_int(ev.get("epochs"), self.planned_epochs)
            if self.epoch_ends and self.epoch_ends[-1].get("epoch") == ev.get("epoch"):
                self.epoch_ends[-1] = ev        # re-emitted epoch: keep the latest
            else:
                self.epoch_ends.append(ev)
        elif kind == "early_stop":
            self.early_stop = ev
        elif kind == "run_end":
            self.run_end = ev
        elif kind == "heartbeat":
            pass                                # only proves the trainer is alive
        elif kind == "system":
            self.systems.append(ev)
        elif kind == "layers":
            self.layers.append(ev)
        elif kind == "histograms":
            epoch = _as_int(ev.get("epoch"), None)
            if epoch is not None:
                self.histograms[epoch] = ev     # one per epoch: the latest wins
                while len(self.histograms) > MAX_HISTOGRAMS:
                    del self.histograms[min(self.histograms)]
        elif kind == "monitor_start":
            self.monitor_start = ev
        elif kind == "inspection":
            self.inspections.append(ev)
        else:
            self.n_ignored += 1                 # unknown type: forward compatibility
            return
        self.n_events += 1
        t = ev.get("time")
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            self.last_event_time = float(t)

    # ── derived state ──────────────────────────────────────────────────────────

    def status(self, now: float | None = None,
               stalled_after_s: float = DEFAULT_STALLED_AFTER_S) -> str:
        """running | completed | early_stopped | failed | stalled | waiting.

        ``stalled``: no ``run_end`` and nothing written for ``stalled_after_s`` (file
        mtime, falling back to the last event time) — the run was probably killed.
        ``waiting``: file exists but holds no event yet.
        """
        if self.run_end is not None:
            end = self.run_end.get("status")
            return end if end in FINAL_STATES else COMPLETED
        last_write = self.mtime if self.mtime is not None else self.last_event_time
        now = time.time() if now is None else now
        if last_write is not None and now - last_write > stalled_after_s:
            return STALLED
        return RUNNING if self.n_events else WAITING

    @property
    def epochs_done(self) -> int:
        return len(self.epoch_ends)

    @property
    def progress(self) -> float | None:
        """Fraction of the planned epochs completed (0..1), None if unknown."""
        if not self.planned_epochs:
            return None
        return min(self.epochs_done / self.planned_epochs, 1.0)

    @property
    def mean_epoch_time_s(self) -> float | None:
        times = [float(e["epoch_time_s"]) for e in self.epoch_ends
                 if isinstance(e.get("epoch_time_s"), (int, float))]
        return sum(times) / len(times) if times else None

    def eta_s(self, now: float | None = None,
              stalled_after_s: float = DEFAULT_STALLED_AFTER_S) -> float | None:
        """Mean epoch time × epochs left, only while the run is going.

        An upper bound when early stopping can end the run sooner.
        """
        if self.status(now, stalled_after_s) not in (RUNNING, WAITING):
            return None
        mean = self.mean_epoch_time_s
        if mean is None or not self.planned_epochs:
            return None
        return mean * max(self.planned_epochs - self.epochs_done, 0)

    @property
    def select_metric(self) -> str | None:
        return (self.run_start or {}).get("select_metric")

    @property
    def best(self) -> float | None:
        """Best value of the selection metric so far (``best`` of the latest epoch_end)."""
        for ev in (self.run_end, *reversed(self.epoch_ends)):
            b = (ev or {}).get("best")
            if isinstance(b, (int, float)) and not isinstance(b, bool):
                return float(b)
        return None

    @property
    def last_val(self) -> dict[str, Any]:
        return (self.epoch_ends[-1].get("val") or {}) if self.epoch_ends else {}

    def epoch_rows(self) -> list[dict[str, Any]]:
        """One flat row per finished epoch: epoch, train_<m>, val_<m>, epoch_time_s, best…"""
        rows = []
        for ev in self.epoch_ends:
            row: dict[str, Any] = {"epoch": ev.get("epoch")}
            for split in ("train", "val"):
                for k, v in (ev.get(split) or {}).items():
                    row[f"{split}_{k}"] = v
            for k in ("epoch_time_s", "best", "is_best"):
                row[k] = ev.get(k)
            rows.append(row)
        return rows

    def batch_rows(self) -> list[dict[str, Any]]:
        """One flat row per kept batch event, with ``progress`` = fractional epoch."""
        rows = []
        for ev in self.batches:
            ep, b, n = ev.get("epoch"), ev.get("batch"), ev.get("n_batches")
            row = dict(ev.get("metrics") or {})
            row.update(epoch=ep, batch=b, n_batches=n)
            if all(isinstance(x, (int, float)) for x in (ep, b, n)) and n:
                row["progress"] = (ep - 1) + b / n
            rows.append(row)
        return rows


def _as_int(value: Any, default: int | None) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        else default


class EventsFollower:
    """Tails one events file by byte offset and keeps the folded :class:`RunState`.

    Each :meth:`poll` reads only what was appended since the last one. Only complete
    lines advance the offset: a half-written last line is left for the next poll
    (unless it already parses as a whole event, i.e. the writer only lacks the
    trailing newline).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.offset = 0
        self.state = RunState()

    def reset(self) -> None:
        self.offset = 0
        self.state = RunState()

    def poll(self) -> int:
        """Reads new bytes and folds them; returns the number of events applied."""
        try:
            st = self.path.stat()
        except OSError:
            return 0
        if st.st_size < self.offset:        # truncated or replaced: start over
            self.reset()
        self.state.mtime = st.st_mtime
        if st.st_size == self.offset:
            return 0
        try:
            with self.path.open("rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return 0

        before = self.state.n_events
        pos = 0
        while pos < len(chunk):
            nl = chunk.find(b"\n", pos)
            complete = nl != -1
            line = chunk[pos:nl] if complete else chunk[pos:]
            if not line.strip():
                pos = nl + 1 if complete else len(chunk)
                continue
            ev = parse_event(line)
            if complete:
                pos = nl + 1
            elif ev is not None:            # whole event, only the "\n" is missing
                pos = len(chunk)
            else:                           # half-written line: retry next poll
                break
            if ev is None:
                self.state.n_ignored += 1
            else:
                self.state.apply(ev)
        self.offset += pos
        return self.state.n_events - before


# ── Discovery ───────────────────────────────────────────────────────────────────

@dataclass
class EventsFile:
    """An ``events_*.jsonl`` found under logs/, with the run it belongs to."""

    timestamp: str
    path: Path
    env: str = "local"
    mode: str = "single"
    model: str = ""
    dataset: str = _DEFAULT_DATASET
    mtime: float = 0.0
    finished: bool = False       # a run_end line exists
    active: bool = False         # not finished and written to recently

    @property
    def label(self) -> str:
        """Same compact label the sidebar uses for the run."""
        return RunInfo(timestamp=self.timestamp, log_path=self.path, trace_mode="simple",
                       env=self.env, mode=self.mode, model=self.model,
                       dataset=self.dataset).label


def has_run_end(path: Path) -> bool:
    """True if the end of the file holds a ``run_end`` event (reads only the tail)."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(size - _TAIL_BYTES, 0))
            tail = f.read()
    except OSError:
        return False
    for line in tail.splitlines():
        if b'"run_end"' in line:
            ev = parse_event(line)
            if ev is not None and ev["type"] == "run_end":
                return True
    return False


def discover_event_files(root: Path = Path("."), now: float | None = None,
                         stalled_after_s: float = DEFAULT_STALLED_AFTER_S) -> list[EventsFile]:
    """Finds every ``events_*.jsonl`` under ``<root>/logs``, active runs first.

    Within each group the newest run (by its timestamp) comes first. A run is
    *active* if its file has no ``run_end`` and was modified within ``stalled_after_s``.
    """
    logs_root = root / "logs"
    if not logs_root.exists():
        return []
    now = time.time() if now is None else now
    found: list[EventsFile] = []
    for path in logs_root.rglob("events_*.jsonl"):
        m = _TIMESTAMP_RE.search(path.stem)
        if not m:
            continue
        env, mode, model, dataset = _env_mode_model_from_path(path, logs_root)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        finished = has_run_end(path)
        found.append(EventsFile(
            timestamp=m.group(1), path=path, env=env, mode=mode, model=model,
            dataset=dataset, mtime=mtime, finished=finished,
            active=not finished and now - mtime <= stalled_after_s))
    found.sort(key=lambda e: timestamp_sort_key(e.timestamp), reverse=True)
    found.sort(key=lambda e: not e.active)      # stable: active first
    return found
