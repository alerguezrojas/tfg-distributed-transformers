"""Manifest of a run: ``run_<timestamp>.json`` next to its ``train_<timestamp>.log``.

A small JSON file (a few KB) with what the ``run_start`` event of the event stream
carries — dataset, task, classes, model, strategy, selection metric and the full
config — plus the command line. The event stream (``events_*.jsonl``) is too big to
commit (the pre-commit hook rejects >2 MB), so the manifest is what lets the web
know, for any committed run, which dataset and metric it is about. Schema in
``docs/eventos.md``.

Pure (stdlib only, no torch): shared by the trainer (``JsonlEventSink``), the web and
``scripts/backfill_run_manifests.py``.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Version of the manifest schema (docs/eventos.md). Adding keys keeps it.
MANIFEST_VERSION = 1

# Keys of the ``run_start`` event copied into the manifest (the rest of the event —
# v/type/time — is bookkeeping of the stream).
_RUN_START_KEYS = ("run_id", "dataset", "task", "classes", "model", "mode", "world_size",
                   "select_metric", "ordinal", "config", "env", "epochs")

_CONFIG_LINE = "Configuración:"


def manifest_path(directory: str | Path, run_id: str) -> Path:
    return Path(directory) / f"run_{run_id}.json"


def build_manifest(run_start: Mapping, *, command: str | None = None,
                   source: str = "training") -> dict:
    """Manifest from the fields of a ``run_start`` event (or of the sink's ``run_info``).

    ``source`` says where it comes from: ``training`` (written by the run itself),
    ``events`` (backfilled from ``events_*.jsonl``) or ``log`` (backfilled from the
    ``Configuración:`` line of the log, with whatever that line records).
    """
    out: dict[str, Any] = {"v": MANIFEST_VERSION, "type": "run_manifest", "source": source}
    out.update({k: run_start[k] for k in _RUN_START_KEYS if k in run_start})
    for extra in ("n_train", "n_val"):          # only the log line knows the split sizes
        if extra in run_start:
            out[extra] = run_start[extra]
    if command:
        out["command"] = command
    return out


def write_manifest(path: str | Path, manifest: Mapping) -> Path:
    """Write atomically (temp file + rename): a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)        # only left behind if writing or renaming failed
    return path


def read_manifest(path: str | Path | None) -> dict | None:
    """The manifest at ``path``, or None if missing, unreadable or not a v1 manifest."""
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("type") != "run_manifest":
        return None
    v = data.get("v", MANIFEST_VERSION)
    if not isinstance(v, int) or isinstance(v, bool) or v > MANIFEST_VERSION:
        return None
    return data


# ── Backfill sources ──────────────────────────────────────────────────────────

def read_run_start(events_path: str | Path | None) -> dict | None:
    """The ``run_start`` event of an ``events_*.jsonl`` (its first line), or None."""
    if events_path is None:
        return None
    try:
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                ev = json.loads(line)
                if isinstance(ev, dict) and ev.get("type") == "run_start":
                    return ev
                break
    except (OSError, ValueError):
        pass
    return None


def parse_config_line(text: str) -> dict[str, str]:
    """``k=v | k=v`` fields of the log's ``Configuración:`` line (``{}`` if absent).

    Values keep their text: ``batch`` is ``16/GPU (global=16)``, ``loss`` can carry
    its parameters (``ce_emd(w=1.0)``).
    """
    for line in text.splitlines():
        i = line.find(_CONFIG_LINE)
        if i < 0:
            continue
        out: dict[str, str] = {}
        for part in line[i + len(_CONFIG_LINE):].split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
        return out
    return {}


def read_config_line(log_path: str | Path, head_bytes: int = 16384) -> dict[str, str]:
    """:func:`parse_config_line` on the top of a log (the line is written at the start)."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            return parse_config_line(f.read(head_bytes))
    except OSError:
        return {}


_LEADING_INT = re.compile(r"\d+")


def _int(text: str | None) -> int | None:
    m = _LEADING_INT.match(text or "")
    return int(m.group()) if m else None


def _float(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def loss_name(text: str | None) -> str | None:
    """``ce_emd(w=1.0)`` → ``ce_emd``."""
    return text.split("(")[0].strip().lower() or None if text else None


def manifest_from_log(log_path: str | Path, *, dataset: str, model: str, mode: str,
                      run_id: str, env: str | None = None,
                      task: str | None = None, select_metric: str | None = None,
                      classes: list[str] | None = None) -> dict:
    """Best-effort manifest of a run that has no event stream, from its log line.

    ``dataset``, ``model`` and ``mode`` come from the folder layout (the caller knows
    it); the ``Configuración:`` line adds the seed, loss, precision, batch, lr,
    epochs and the train/val sizes — all that older runs recorded. Unknown fields are
    left out, never invented.
    """
    cfg = read_config_line(log_path)
    train: dict[str, Any] = {}
    for key, name, conv in (("epochs", "epochs", _int), ("lr", "lr", _float),
                            ("seed", "seed", _int), ("precision", "precision", str)):
        value = conv(cfg.get(key)) if key in cfg and cfg[key] != "None" else None
        if value is not None:
            train[name] = value
    if (bs := _int(cfg.get("batch"))) is not None:
        train["batch_size"] = bs        # per GPU
    if (loss := loss_name(cfg.get("loss"))):
        train["loss"] = loss
    if select_metric:
        train["select_by"] = select_metric
    config: dict[str, Any] = {"data": {"name": cfg.get("dataset", dataset)},
                              "model": {"name": cfg.get("modelo", model)},
                              "training": train}
    if env:
        config["output"] = {"env": env}
    run_start: dict[str, Any] = {"run_id": run_id, "dataset": cfg.get("dataset", dataset),
                                 "model": cfg.get("modelo", model), "mode": mode,
                                 "config": config}
    if task:
        run_start["task"] = task
    if classes:
        run_start["classes"] = classes
    if select_metric:
        run_start["select_metric"] = select_metric
    if "epochs" in train:
        run_start["epochs"] = train["epochs"]
    for src, dst in (("train", "n_train"), ("val", "n_val")):
        if (n := _int(cfg.get(src))) is not None:
            run_start[dst] = n
    return build_manifest(run_start, source="log")
