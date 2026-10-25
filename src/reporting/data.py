"""Run resolution and event loading for the report (read-only, CPU, no torch).

``resolve_run`` turns what the user typed (an ``events_*.jsonl``, a run folder or a
``DDMMYYYY_HHMMSS`` stamp) into the events file; ``load_run`` folds that file into a
:class:`RunData` with every event type the report uses, without the in-memory caps of the web's
live follower (a report is built once, from the whole file).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.web.events_reader import parse_event
from src.web.inspector_data import resolve_inspect_dir

_STAMP = re.compile(r"^\d{8}_\d{6}$")
_STAMP_IN = re.compile(r"(\d{8}_\d{6})")
_STAMP_KEY = re.compile(r"(\d{2})(\d{2})(\d{4})_(\d{6})")


class RunNotFound(ValueError):
    """The run argument does not point to exactly one events file."""


def _newest(paths: list[Path]) -> Path:
    """Newest by the DDMMYYYY_HHMMSS stamp of the name (not by mtime or alphabetically)."""
    def key(p: Path):
        m = _STAMP_KEY.search(p.stem)
        return (m.group(3), m.group(2), m.group(1), m.group(4)) if m else ("",) * 4
    return max(paths, key=key)


def resolve_run(spec: str | Path, root: Path | str = ".") -> Path:
    """Path of the ``events_*.jsonl`` that ``spec`` designates.

    * a file: an ``events_*.jsonl`` itself, or any ``<kind>_<stamp>.<ext>`` of the run (the
      events file with the same stamp next to it is used);
    * a folder: the run folder (the newest ``events_*.jsonl`` directly inside, else below it),
      or the ``inspect_<stamp>`` directory of the run;
    * a stamp ``DDMMYYYY_HHMMSS``: searched under ``<root>/logs`` (error if it is in several places).
    """
    root = Path(root)
    text = str(spec)
    path = Path(text)
    if path.is_file():
        if path.name.startswith("events_") and path.suffix == ".jsonl":
            return path
        m = _STAMP_IN.search(path.name)
        sibling = path.parent / f"events_{m.group(1)}.jsonl" if m else None
        if sibling is not None and sibling.is_file():
            return sibling
        raise RunNotFound(f"{path} is not an events file and has no events_<stamp>.jsonl next to it")
    if path.is_dir():
        if path.name.startswith("inspect_"):
            m = _STAMP_IN.search(path.name)
            sibling = path.parent / f"events_{m.group(1)}.jsonl" if m else None
            if sibling is not None and sibling.is_file():
                return sibling
        found = list(path.glob("events_*.jsonl")) or list(path.rglob("events_*.jsonl"))
        if not found:
            raise RunNotFound(f"no events_*.jsonl in {path} (was the run trained with --trace "
                              "other than off?)")
        return _newest(found)
    if _STAMP.match(text):
        found = sorted((root / "logs").rglob(f"events_{text}.jsonl"))
        if not found:
            raise RunNotFound(f"no events_{text}.jsonl under {root / 'logs'}")
        if len(found) > 1:
            raise RunNotFound(f"stamp {text} is ambiguous: " + ", ".join(str(p) for p in found))
        return found[0]
    raise RunNotFound(f"{text!r} is not a file, a folder or a DDMMYYYY_HHMMSS stamp")


def stamp_of(events_path: Path) -> str:
    m = _STAMP_IN.search(events_path.stem)
    return m.group(1) if m else events_path.stem


@dataclass
class RunData:
    """Everything the report reads from the events file, in order."""
    events_path: Path
    stamp: str
    run_start: dict[str, Any] | None = None
    monitor_start: dict[str, Any] | None = None
    epoch_ends: list[dict[str, Any]] = field(default_factory=list)
    batches: list[dict[str, Any]] = field(default_factory=list)
    systems: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    early_stop: dict[str, Any] | None = None
    run_end: dict[str, Any] | None = None
    first_time: float | None = None
    last_time: float | None = None
    n_events: int = 0
    inspect_dir: Path | None = None

    @property
    def task(self) -> str | None:
        return (self.run_start or {}).get("task")

    @property
    def config(self) -> dict[str, Any]:
        cfg = (self.run_start or {}).get("config")
        return cfg if isinstance(cfg, dict) else {}


def load_run(events_path: Path | str) -> RunData:
    """Fold an events file (unknown types and newer schema versions are ignored; a last
    incomplete line is dropped, as every reader of the stream must do)."""
    path = Path(events_path)
    run = RunData(events_path=path, stamp=stamp_of(path))
    with path.open("rb") as f:
        lines = f.read().split(b"\n")
    for raw in lines:
        if not raw.strip():
            continue
        ev = parse_event(raw)
        if ev is None:
            continue
        run.n_events += 1
        t = ev.get("time")
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            run.first_time = t if run.first_time is None else run.first_time
            run.last_time = t
        _fold(run, ev)
    if run.n_events == 0:
        raise RunNotFound(f"{path} has no usable events")
    inspect = resolve_inspect_dir(path, run.monitor_start)
    if inspect is None:
        guess = path.parent / f"inspect_{run.stamp}"
        inspect = guess if guess.is_dir() else None
    run.inspect_dir = inspect
    return run


def _fold(run: RunData, ev: dict[str, Any]) -> None:
    kind = ev["type"]
    if kind == "epoch_end":
        if run.epoch_ends and run.epoch_ends[-1].get("epoch") == ev.get("epoch"):
            run.epoch_ends[-1] = ev          # re-emitted epoch: keep the latest
        else:
            run.epoch_ends.append(ev)
        return
    lists = {"batch": run.batches, "system": run.systems, "alert": run.alerts}
    if kind in lists:
        lists[kind].append(ev)
        return
    single = {"run_start": "run_start", "monitor_start": "monitor_start",
              "early_stop": "early_stop", "run_end": "run_end"}
    if kind in single:
        setattr(run, single[kind], ev)
