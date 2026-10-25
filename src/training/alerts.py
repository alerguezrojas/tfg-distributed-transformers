"""Training alerts — rules that turn the event stream into ``alert`` events.

Two layers (contract: ``docs/inspeccion.md``):

* **Rules** (``check_*``): pure functions of plain numbers, no torch, no I/O. Each
  returns a :class:`Finding` when the condition holds, ``None`` otherwise.
* :class:`AlertMonitor`: a :class:`~src.training.events.TrainingListener` that feeds the
  rules from the epoch results, the batch hook and the events other producers emit
  (``system``, ``layers``: it subscribes to the sink) and writes an ``alert`` event for
  each new finding. An alert is emitted once per ``(code, group)`` and re-armed only after
  the condition has cleared.

Observers never stop or change the training: every entry point catches its own errors
(one warning), nothing here touches torch, the model or the global RNG.
"""
from __future__ import annotations

import math
import statistics
import threading
import warnings
from collections import deque
from typing import NamedTuple

from src.training.events import TrainingListener

#: Defaults of ``monitor.alerts`` (docs/inspeccion.md). ``enabled`` is on whenever the
#: event stream exists: the rules are a few comparisons per epoch / sample.
ALERT_DEFAULTS: dict = {
    "enabled": True,
    "loss_divergence_factor": 2.0,   # val loss > k × the best so far
    "overfit_epochs": 3,             # val loss rising N epochs in a row, train falling
    "grad_explosion_factor": 100.0,  # group grad norm > k × its recent median
    "grad_history": 20,              # recent `layers` samples kept per group
    "update_ratio_min": 1e-7,        # LLRD + LR decay make low groups ~1e-6
    "update_ratio_max": 1e-1,
    "update_ratio_patience": 3,      # consecutive `layers` samples out of range
    "data_wait_frac": 0.3,           # share of the epoch waiting for data
    "gpu_mem_pct": 95.0,
    "gpu_temp_c": 85.0,
    "throughput_drop_factor": 0.5,   # epoch samples/s < k × median of the previous ones
}

#: Samples of history a median-based rule needs before it can fire.
_MIN_GRAD_HISTORY = 5
#: ``group`` of the ``grad_explosion`` alerts about the whole model's gradient norm.
TOTAL_GROUP = "total"
_MIN_THROUGHPUT_HISTORY = 2


class Finding(NamedTuple):
    """A rule fired. ``group`` tells apart the instances of one code (layer group,
    GPU, split); ``None`` for global conditions."""
    code: str
    level: str
    message: str
    value: float | None = None
    threshold: float | None = None
    group: str | None = None


def alerts_config(cfg: dict) -> dict:
    """The ``monitor.alerts`` section of a config with every default filled in."""
    return {**ALERT_DEFAULTS, **((cfg.get("monitor") or {}).get("alerts") or {})}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ── Rules (pure) ──────────────────────────────────────────────────────────────

def check_nan_loss(value, group: str = "train", where: str = "epoch") -> Finding | None:
    """Critical: a loss that is NaN or infinite. ``None`` (unknown) is not a finding."""
    if value is None or _finite(value):
        return None
    return Finding("nan_loss", "critical",
                   f"The {group} loss is {value} ({where}); training has diverged.",
                   None, None, group)


def check_loss_divergence(val_losses: list[float], factor: float = 2.0) -> Finding | None:
    """Warning: the latest validation loss is more than ``factor`` × the best before it."""
    if len(val_losses) < 2:
        return None
    best, last = min(val_losses[:-1]), val_losses[-1]
    if best <= 0 or not (_finite(best) and _finite(last)):
        return None
    if last > factor * best:
        return Finding("loss_divergence", "warning",
                       f"Validation loss {last:.4g} is {last / best:.1f}x the best so far "
                       f"({best:.4g}).", last, factor * best)
    return None


def check_overfitting(train_losses: list[float], val_losses: list[float],
                      epochs: int = 3) -> Finding | None:
    """Warning: validation loss rose in each of the last ``epochs`` epochs while the
    training loss fell in each of them (needs ``epochs + 1`` points of both)."""
    n = epochs + 1
    if epochs < 1 or len(val_losses) < n or len(train_losses) < n:
        return None
    val, train = val_losses[-n:], train_losses[-n:]
    if not all(_finite(x) for x in val + train):
        return None
    rising = all(b > a for a, b in zip(val, val[1:]))
    falling = all(b < a for a, b in zip(train, train[1:]))
    if rising and falling:
        return Finding("overfitting", "warning",
                       f"Validation loss rose for {epochs} epochs in a row "
                       f"({val[0]:.4g} -> {val[-1]:.4g}) while the training loss fell "
                       f"({train[0]:.4g} -> {train[-1]:.4g}).", val[-1], float(epochs))
    return None


def check_grad_explosion(name: str, norm, history: list[float],
                         factor: float = 100.0) -> Finding | None:
    """Warning: a group's gradient norm exceeds ``factor`` × the median of its recent norms.

    ``history`` are the previous norms of that group (not including ``norm``)."""
    past = [h for h in history if _finite(h)]
    if len(past) < _MIN_GRAD_HISTORY:
        return None
    median = statistics.median(past)
    if median <= 0:
        return None
    if not _finite(norm):
        return Finding("grad_explosion", "warning",
                       f"Gradient norm of {name} is {norm} (recent median {median:.3g}).",
                       None, factor * median, name)
    if norm > factor * median:
        return Finding("grad_explosion", "warning",
                       f"Gradient norm of {name} is {norm:.3g}, {norm / median:.0f}x its "
                       f"recent median ({median:.3g}).", norm, factor * median, name)
    return None


def check_update_ratio(name: str, ratios: list, lo: float = 1e-7, hi: float = 1e-1,
                       patience: int = 3) -> Finding | None:
    """Warning: the update/weight ratio of a group was outside ``[lo, hi]`` in each of the
    last ``patience`` samples (``None`` samples count as in range)."""
    if patience < 1 or len(ratios) < patience:
        return None
    last = ratios[-patience:]
    if not all(_finite(r) and not lo <= r <= hi for r in last):
        return None
    ratio = last[-1]
    bound = lo if ratio < lo else hi
    return Finding("update_ratio_out_of_range", "warning",
                   f"Update/weight ratio of {name} is {ratio:.3g}, outside "
                   f"[{lo:g}, {hi:g}] for {patience} samples in a row.", ratio, bound, name)


def check_dataloader_bound(frac, threshold: float = 0.3) -> Finding | None:
    """Warning: the epoch spent more than ``threshold`` of its time waiting for data."""
    if _finite(frac) and frac > threshold:
        return Finding("dataloader_bound", "warning",
                       f"Training waited for data {frac:.0%} of the epoch (limit "
                       f"{threshold:.0%}): the DataLoader is the bottleneck.",
                       frac, threshold)
    return None


def check_gpu_memory(index, used_mb, total_mb, threshold_pct: float = 95.0) -> Finding | None:
    """Warning: GPU memory in use above ``threshold_pct`` % of the total."""
    if not (_finite(used_mb) and _finite(total_mb)) or total_mb <= 0:
        return None
    pct = 100.0 * used_mb / total_mb
    if pct > threshold_pct:
        return Finding("gpu_memory_high", "warning",
                       f"GPU {index} memory is at {pct:.1f}% ({used_mb:.0f} of "
                       f"{total_mb:.0f} MiB); an out-of-memory error is close.",
                       pct, threshold_pct, f"gpu{index}")
    return None


def check_gpu_temp(index, temp_c, threshold_c: float = 85.0) -> Finding | None:
    """Warning: GPU temperature above ``threshold_c`` degrees Celsius."""
    if _finite(temp_c) and temp_c > threshold_c:
        return Finding("gpu_temp_high", "warning",
                       f"GPU {index} is at {temp_c:.0f} C (limit {threshold_c:.0f} C); "
                       f"it may throttle.", float(temp_c), threshold_c, f"gpu{index}")
    return None


def check_throughput_drop(history: list[float], current,
                          factor: float = 0.5) -> Finding | None:
    """Warning: this epoch's samples/s is below ``factor`` × the median of the previous
    epochs (needs at least two)."""
    past = [h for h in history if _finite(h) and h > 0]
    if len(past) < _MIN_THROUGHPUT_HISTORY or not _finite(current):
        return None
    median = statistics.median(past)
    if current < factor * median:
        return Finding("throughput_drop", "warning",
                       f"Throughput fell to {current:.1f} samples/s, "
                       f"{current / median:.0%} of the previous median ({median:.1f}).",
                       current, factor * median)
    return None


# ── Monitor ───────────────────────────────────────────────────────────────────

class AlertMonitor(TrainingListener):
    """Evaluates the rules and emits ``alert`` events through ``sink.emit``.

    Feeds: ``on_epoch_end`` (losses, ``data_wait_frac``, ``samples_per_s``), ``on_batch``
    (a batch hook, only worth registering when the Trainer already computes per-batch
    metrics) and ``on_event`` (the sink calls it for ``system`` and ``layers`` events, so
    the sampler and the layer monitor know nothing about alerts). ``grad_explosion`` and
    ``update_ratio_out_of_range`` are only evaluated if ``layers`` events exist.
    """

    def __init__(self, sink, config: dict | None = None, grad_norm_source=None):
        self._sink = sink
        self._grad_norm_source = grad_norm_source   # () -> pre-clip total grad norm
        self._cfg = {**ALERT_DEFAULTS, **(config or {})}
        self._lock = threading.RLock()      # system sampler thread vs. training thread
        self._active: set[tuple[str, str | None]] = set()
        self._epoch: int | None = None
        self._pairs: list[tuple[float, float]] = []    # (train, val) of finite epochs
        self._val: list[float] = []
        self._throughput: list[float] = []
        self._grads: dict[str, deque] = {}
        self._ratios: dict[str, deque] = {}
        self._loss_finite: bool | None = None   # of the latest batch (None: no batch hook)
        self._warned = False
        self.emitted: list[dict] = []       # alerts written so far (for tests / summaries)

    # ── core ─────────────────────────────────────────────────────────────────

    def _warn(self, what: str, exc: BaseException) -> None:
        if not self._warned:
            self._warned = True
            warnings.warn(f"alert monitor: {what} ({exc!r}); the training continues",
                          RuntimeWarning, stacklevel=3)

    def _update(self, code: str, findings: list[Finding], epoch, batch=None,
                scope=lambda group: True) -> None:
        """Emit the findings not already active for ``code``; forget (re-arm) the active
        groups in ``scope`` that are no longer among the findings (condition cleared).
        A pair is marked active only after its event was written, so a failed write is
        retried on the next evaluation."""
        now = {f.group: f for f in findings}
        with self._lock:
            self._active -= {key for key in self._active
                             if key[0] == code and key[1] not in now and scope(key[1])}
            for group, f in now.items():
                if (code, group) in self._active:
                    continue
                fields = {"level": f.level, "code": f.code, "message": f.message,
                          "epoch": epoch, "batch": batch, "value": f.value,
                          "threshold": f.threshold, "group": f.group}
                self._sink.emit("alert", **fields)
                self._active.add((code, group))
                self.emitted.append(fields)

    # ── TrainingListener ─────────────────────────────────────────────────────

    def on_epoch_start(self, epoch, epochs):
        self._epoch = epoch

    def on_epoch_end(self, epoch, epochs, train_m, val_m, best, is_best, epoch_time_s):
        try:
            self._epoch_end(epoch, train_m, val_m)
        except Exception as exc:  # noqa: BLE001 — never stop the training
            self._warn(f"epoch {epoch} rules failed", exc)

    def _epoch_end(self, epoch, train_m, val_m) -> None:
        cfg = self._cfg
        train_loss, val_loss = train_m.get("loss"), val_m.get("loss")
        with self._lock:
            self._update("nan_loss", [f for f in (
                check_nan_loss(train_loss, "train"), check_nan_loss(val_loss, "val")) if f],
                epoch)
            if _finite(val_loss):
                self._val.append(float(val_loss))
            if _finite(val_loss) and _finite(train_loss):
                self._pairs.append((float(train_loss), float(val_loss)))
            self._update("loss_divergence", [f for f in [check_loss_divergence(
                self._val, cfg["loss_divergence_factor"])] if f], epoch)
            self._update("overfitting", [f for f in [check_overfitting(
                [t for t, _ in self._pairs], [v for _, v in self._pairs],
                int(cfg["overfit_epochs"]))] if f], epoch)
            self._update("dataloader_bound", [f for f in [check_dataloader_bound(
                train_m.get("data_wait_frac"), cfg["data_wait_frac"])] if f], epoch)
            sps = train_m.get("samples_per_s")
            self._update("throughput_drop", [f for f in [check_throughput_drop(
                self._throughput, sps, cfg["throughput_drop_factor"])] if f], epoch)
            if _finite(sps):
                self._throughput.append(float(sps))

    def on_batch(self, epoch: int, batch_idx: int, n_batches: int, metrics: dict) -> None:
        """Batch hook: NaN/inf loss in a batch and spikes of the total gradient norm
        (``grad_norm_source``: the norm before clipping). Unguarded by the Trainer, so it catches."""
        try:
            loss = metrics.get("batch_loss")
            self._loss_finite = _finite(loss)
            f = check_nan_loss(loss, "train", f"batch {batch_idx}")
            if f is not None:   # only adds: re-arming is decided at the end of the epoch
                self._update("nan_loss", [f], self._epoch or epoch, batch_idx,
                             scope=lambda group: False)
            if self._grad_norm_source is not None:
                self._total_grad(self._epoch or epoch, batch_idx, self._grad_norm_source())
        except Exception as exc:  # noqa: BLE001
            self._warn("batch rule failed", exc)

    def _total_grad(self, epoch, batch_idx, norm) -> None:
        if norm is None:
            return
        with self._lock:
            if self._skipped_step(norm):
                return
            f = self._grad_finding(TOTAL_GROUP, norm)
            self._update("grad_explosion", [f] if f else [], epoch, batch_idx,
                         scope=lambda group: group == TOTAL_GROUP)

    def _skipped_step(self, norm) -> bool:
        """A non-finite gradient with a finite loss is a step the GradScaler skips (fp16
        overflow): routine, not an explosion."""
        return not _finite(norm) and self._loss_finite is True

    # ── Events from other producers (sink subscription) ──────────────────────

    def on_event(self, event_type: str, fields: dict) -> None:
        """Called by the sink after each ``emit``; ignores everything but the feeds."""
        if event_type not in ("system", "layers"):
            return
        try:
            if event_type == "system":
                self._system(fields)
            else:
                self._layers(fields)
        except Exception as exc:  # noqa: BLE001
            self._warn(f"{event_type} rules failed", exc)

    def _system(self, fields: dict) -> None:
        mem, temp = [], []
        for gpu in fields.get("gpus") or []:
            idx = gpu.get("index")
            mem.append(check_gpu_memory(idx, gpu.get("mem_used_mb"), gpu.get("mem_total_mb"),
                                        self._cfg["gpu_mem_pct"]))
            temp.append(check_gpu_temp(idx, gpu.get("temp_c"), self._cfg["gpu_temp_c"]))
        with self._lock:
            epoch = self._epoch
            self._update("gpu_memory_high", [f for f in mem if f], epoch)
            self._update("gpu_temp_high", [f for f in temp if f], epoch)

    def _grad_finding(self, name: str, norm) -> Finding | None:
        """Explosion check of one group (caller holds the lock); updates its history."""
        if self._skipped_step(norm):
            return None
        hist = self._grads.setdefault(name, deque(maxlen=int(self._cfg["grad_history"])))
        finding = check_grad_explosion(name, norm, list(hist),
                                       self._cfg["grad_explosion_factor"])
        if _finite(norm):
            hist.append(float(norm))
        return finding

    def _ratio_finding(self, name: str, ratio) -> Finding | None:
        cfg = self._cfg
        patience = max(1, int(cfg["update_ratio_patience"]))
        hist = self._ratios.setdefault(name, deque(maxlen=patience))
        hist.append(ratio)
        return check_update_ratio(name, list(hist), cfg["update_ratio_min"],
                                  cfg["update_ratio_max"], patience)

    def _layers(self, fields: dict) -> None:
        epoch, batch = fields.get("epoch"), fields.get("batch")
        with self._lock:
            rows = fields.get("groups") or []
            grads = [self._grad_finding(r.get("name"), r.get("grad_norm")) for r in rows]
            ratios = [self._ratio_finding(r.get("name"), r.get("update_ratio")) for r in rows]
            self._update("grad_explosion", [f for f in grads if f], epoch, batch,
                         scope=lambda group: group != TOTAL_GROUP)
            self._update("update_ratio_out_of_range", [f for f in ratios if f], epoch, batch)
