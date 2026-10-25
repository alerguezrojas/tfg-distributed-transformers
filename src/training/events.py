"""Training events — a structured, append-only record of each run.

:class:`TrainingListener` is the Observer interface of the training loop: an
``EpochController`` notifies every subscribed listener (``add_listener``) when the
run starts, an epoch starts or ends, early stopping fires and the run ends or
fails. Listeners only observe; they never change the training.

:class:`JsonlEventSink` is the listener that writes ``events_<timestamp>.jsonl``:
one JSON object per line, flushed immediately, so the web dashboard can follow a
run while it trains. The schema (version :data:`SCHEMA_VERSION`) is the contract
with its readers and is documented in ``docs/eventos.md``. Only rank 0 writes it.
"""
from __future__ import annotations

import json
import math
import platform
import socket
import subprocess
import threading
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Version of the event schema (docs/eventos.md). Adding fields or event types
#: keeps it; removing or redefining a field bumps it.
SCHEMA_VERSION = 1

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TrainingListener:
    """Observer of the training loop. Every method is an optional no-op hook."""

    def on_fit_start(self, epochs: int) -> None:
        """The run starts; ``epochs`` is the number of planned epochs."""

    def on_epoch_start(self, epoch: int, epochs: int) -> None:
        """Epoch ``epoch`` (1-based) starts."""

    def on_epoch_end(self, epoch: int, epochs: int, train_m: dict, val_m: dict,
                     best: float, is_best: bool, epoch_time_s: float) -> None:
        """Epoch ``epoch`` ended; ``best`` is the best selection metric so far."""

    def on_early_stop(self, epoch: int, best: float) -> None:
        """Early stopping ended the run after ``epoch``."""

    def on_fit_end(self, best: float, epochs_run: int, early_stopped: bool) -> None:
        """The run finished normally (all epochs or early stopping)."""

    def on_fit_error(self, error: BaseException, epochs_run: int, best: float) -> None:
        """The run raised ``error`` after ``epochs_run`` complete epochs."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_value(value: Any) -> Any:
    """Plain JSON value: tensors/NumPy scalars → float, NaN/inf → None."""
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def public_metrics(metrics: Mapping) -> dict:
    """Scalar metrics of a train/eval result, without the private ``_`` tensors.

    The validation threshold tuned by the task (``_optimal_threshold``) is kept
    under its public name, as in the checkpoint.
    """
    out = {k: _json_value(v) for k, v in metrics.items()
           if not k.startswith("_") and getattr(v, "ndim", 0) == 0}
    for private in ("_optimal_threshold", "_f1_at_optimal_threshold"):
        if metrics.get(private) is not None:
            out[private[1:]] = _json_value(metrics[private])
    return out


def _best(value: float) -> float | None:
    """The loop starts ``best`` at -1 (no epoch yet); report that as null."""
    return None if value is None or value < 0 else value


def _git_commit() -> str | None:
    try:
        out = subprocess.run(  # noqa: S603, S607 — fixed command, no user input
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT,
            capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_fingerprint() -> dict:
    """Host, library versions, code version and devices of this process."""
    import torch
    try:
        import timm
        timm_version = timm.__version__
    except ImportError:
        timm_version = None
    if torch.cuda.is_available():
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm_version,
        "git_commit": _git_commit(),
        "devices": devices,
    }


# ── JSONL sink ────────────────────────────────────────────────────────────────

class JsonlEventSink(TrainingListener):
    """Writes the run's events to a JSON Lines file (schema in docs/eventos.md).

    ``run_info`` holds the static ``run_start`` fields (run_id, dataset, task,
    classes, model, mode, world_size, select_metric, config, env); the planned
    number of epochs is added when the run starts.

    While the run is alive a daemon thread writes a ``heartbeat`` every
    ``heartbeat_s`` seconds, so a reader can tell a long epoch from a dead process
    without the training loop doing any extra work.
    """

    def __init__(self, path: str | Path, run_info: Mapping, log_every: int = 1,
                 heartbeat_s: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")            # one file per run (timestamped name)
        self._run_info = dict(run_info)
        self._log_every = max(1, int(log_every))
        self._heartbeat_s = heartbeat_s
        self._t_start: float | None = None
        self._epoch: int | None = None      # loop epoch, from on_epoch_start
        self._lock = threading.Lock()       # heartbeat thread vs. training thread
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        self._batch_warned = False

    def _write(self, event_type: str, **fields) -> None:
        record = {"v": SCHEMA_VERSION, "type": event_type, "time": time.time(),
                  **{k: _json_value(v) for k, v in fields.items()}}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def _beat(self) -> None:
        while not self._stop.wait(self._heartbeat_s):
            try:
                self._write("heartbeat", epoch=self._epoch, elapsed_s=self._elapsed())
            except Exception:  # noqa: BLE001 — a lost heartbeat only delays "stalled"
                pass

    def _start_heartbeat(self) -> None:
        if self._heartbeat_s and self._heartbeat_s > 0:
            self._stop.clear()
            self._heartbeat = threading.Thread(target=self._beat, name="events-heartbeat",
                                               daemon=True)
            self._heartbeat.start()

    def _stop_heartbeat(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=5)
            self._heartbeat = None

    def _elapsed(self) -> float | None:
        return None if self._t_start is None else time.time() - self._t_start

    # ── TrainingListener ─────────────────────────────────────────────────────

    def on_fit_start(self, epochs):
        self._t_start = time.time()
        self._write("run_start", **self._run_info, epochs=epochs)
        self._start_heartbeat()

    def on_epoch_start(self, epoch, epochs):
        self._epoch = epoch
        self._write("epoch_start", epoch=epoch, epochs=epochs)

    def on_epoch_end(self, epoch, epochs, train_m, val_m, best, is_best, epoch_time_s):
        self._write("epoch_end", epoch=epoch, epochs=epochs,
                    train=public_metrics(train_m), val=public_metrics(val_m),
                    epoch_time_s=epoch_time_s, best=_best(best), is_best=is_best)

    def on_early_stop(self, epoch, best):
        self._write("early_stop", epoch=epoch, best=_best(best))

    def on_fit_end(self, best, epochs_run, early_stopped):
        self._stop_heartbeat()
        self._write("run_end", status="early_stopped" if early_stopped else "completed",
                    epochs_run=epochs_run, best=_best(best), total_time_s=self._elapsed())

    def on_fit_error(self, error, epochs_run, best):
        self._stop_heartbeat()
        # Ctrl-C / SystemExit are not Exceptions: an interruption, not a failure.
        status = "failed" if isinstance(error, Exception) else "interrupted"
        self._write("run_end", status=status, epochs_run=epochs_run, best=_best(best),
                    total_time_s=self._elapsed(),
                    error=f"{type(error).__name__}: {error}")

    # ── Batch hook (Trainer.register_batch_hook) ─────────────────────────────

    def on_batch(self, epoch: int, batch_idx: int, n_batches: int, metrics: dict) -> None:
        """Batch hook: one ``batch`` event every ``log_every`` batches (and the last).

        The epoch is the loop's (as in ``epoch_start``/``epoch_end``), not the
        trainer's counter, which continues from the checkpoint after ``--resume``.
        The Trainer calls batch hooks unguarded, so a write error is turned into a
        warning here: the event stream must never stop a training run.
        """
        if batch_idx % self._log_every != 0 and batch_idx != n_batches:
            return
        try:
            self._write("batch", epoch=self._epoch if self._epoch is not None else epoch,
                        batch=batch_idx, n_batches=n_batches, metrics=metrics)
        except Exception as exc:  # noqa: BLE001
            if not self._batch_warned:
                self._batch_warned = True
                warnings.warn(f"events: batch event not written ({exc!r}); further "
                              f"batch write errors are silenced", RuntimeWarning,
                              stacklevel=2)
