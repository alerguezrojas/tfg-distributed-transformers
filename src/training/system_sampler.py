"""System sampler — periodic ``system`` events (GPU, CPU, RAM) during a run.

:class:`SystemSampler` is a :class:`~src.training.events.TrainingListener`: when the
first epoch starts (so no sample can precede ``run_start``) it launches a daemon thread that, every ``interval_s`` seconds, reads the
state of the machine and appends one ``system`` event to the run's event stream
(``sink.emit``). The fields are the contract documented in ``docs/inspeccion.md``.

The thread only *reads* (NVML and psutil); it never touches torch or CUDA, so the
training is unchanged. It is also best-effort: no NVML or no GPU gives ``gpus: []``,
and a failing read or write is turned into one warning — it never stops the run.

GPUs: the ones the process can see, identified by UUID. NVML numbers devices by PCI bus
while CUDA orders them by ``CUDA_DEVICE_ORDER`` (fastest first by default), so an NVML index
is not ``cuda:N``. The builder passes the UUIDs of torch's visible devices
(:func:`visible_gpu_uuids`) and the probe resolves them in that order: ``index`` is then
``cuda:N`` and the order of ``env.devices`` in ``run_start``. Without torch/CUDA it falls back
to NVML's own numbering filtered by ``CUDA_VISIBLE_DEVICES`` (indices, or ``GPU-`` UUIDs;
empty = none), where ``index`` is the position among the visible ones.
"""
from __future__ import annotations

import os
import threading
import warnings

from src.training.events import TrainingListener

_MIB = 1024 ** 2


def _text(value) -> str:
    """NVML returns ``bytes`` in old releases and ``str`` in new ones."""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def gpu_uuid(index: int) -> str | None:
    """NVML-style UUID (``GPU-xxxxxxxx-…``) of torch's ``cuda:<index>``, or ``None``."""
    try:
        import torch
        return f"GPU-{torch.cuda.get_device_properties(index).uuid}"
    except Exception:  # noqa: BLE001 — no torch / no CUDA / torch without .uuid
        return None


def visible_gpu_uuids() -> list[str] | None:
    """UUIDs of the GPUs torch sees, in ``cuda:N`` order (``CUDA_VISIBLE_DEVICES`` and
    ``CUDA_DEVICE_ORDER`` already applied); ``None`` if they cannot be determined."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        uuids = [gpu_uuid(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        return None
    return uuids if uuids and all(uuids) else None


def _safe(fn):
    """Value of ``fn()``, or ``None`` if this GPU does not support the query."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 — e.g. NVML_ERROR_NOT_SUPPORTED on some boards
        return None


class SystemProbe:
    """One reading of the machine's state, in the shape of a ``system`` event.

    ``nvml`` and ``psutil_mod`` are injectable (tests); by default they are imported
    when needed, and a missing or failing NVML just means no GPUs are reported.
    """

    def __init__(self, nvml=None, psutil_mod=None, visible_devices: str | None = None,
                 gpu_uuids: list[str] | None = None):
        self._nvml = nvml
        self._handles: list = []
        self._nvml_up = False
        self._gpu_uuids = gpu_uuids
        self._init_nvml(visible_devices if visible_devices is not None
                        else os.environ.get("CUDA_VISIBLE_DEVICES"))
        if psutil_mod is None:
            import psutil as psutil_mod
        self._psutil = psutil_mod
        self._proc = psutil_mod.Process()
        # psutil's percentages are "since the previous call": prime them so that the
        # first real sample is meaningful.
        psutil_mod.cpu_percent(None)
        self._proc.cpu_percent(None)

    def _init_nvml(self, visible: str | None) -> None:
        try:
            nvml = self._nvml
            if nvml is None:
                import pynvml as nvml
                self._nvml = nvml
            nvml.nvmlInit()
            self._nvml_up = True
            self._handles = self._handles_by_uuid(nvml, self._gpu_uuids) \
                or self._resolve_handles(nvml, visible)
        except Exception:  # noqa: BLE001 — no driver / no GPU / no pynvml → gpus: []
            self._handles = []

    @staticmethod
    def _handles_by_uuid(nvml, uuids: list[str] | None) -> list:
        """Handles of ``uuids`` in order (``cuda:N`` order); ``[]`` if any is unknown."""
        try:
            return [nvml.nvmlDeviceGetHandleByUUID(u) for u in uuids or []]
        except Exception:  # noqa: BLE001 — fall back to the env-var logic
            return []

    @staticmethod
    def _resolve_handles(nvml, visible: str | None) -> list:
        if visible is None:
            return [nvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(nvml.nvmlDeviceGetCount())]
        handles = []
        for token in (t.strip() for t in visible.split(",")):
            try:
                handles.append(nvml.nvmlDeviceGetHandleByIndex(int(token)) if token.isdigit()
                               else nvml.nvmlDeviceGetHandleByUUID(token))
            except Exception:  # noqa: BLE001 — CUDA stops at the first invalid entry
                break
        return handles

    def _gpu(self, index: int, handle) -> dict:
        nvml = self._nvml
        mem = _safe(lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
        power = _safe(lambda: nvml.nvmlDeviceGetPowerUsage(handle))
        return {
            "index": index,
            "uuid": _safe(lambda: _text(nvml.nvmlDeviceGetUUID(handle))),
            "name": _safe(lambda: _text(nvml.nvmlDeviceGetName(handle))),
            "util_pct": _safe(lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu),
            "mem_used_mb": None if mem is None else mem.used / _MIB,
            "mem_total_mb": None if mem is None else mem.total / _MIB,
            "power_w": None if power is None else power / 1000.0,   # NVML gives mW
            "temp_c": _safe(lambda: nvml.nvmlDeviceGetTemperature(
                handle, nvml.NVML_TEMPERATURE_GPU)),
            "sm_clock_mhz": _safe(lambda: nvml.nvmlDeviceGetClockInfo(
                handle, nvml.NVML_CLOCK_SM)),
        }

    def sample(self) -> dict:
        """The fields of one ``system`` event."""
        ps = self._psutil
        return {
            "gpus": [self._gpu(i, h) for i, h in enumerate(self._handles)],
            "cpu_pct": ps.cpu_percent(None),
            "ram_used_pct": ps.virtual_memory().percent,
            # Of this process only (can exceed 100 with several threads); DataLoader
            # workers are separate processes and are not included.
            "proc_cpu_pct": self._proc.cpu_percent(None),
            "proc_rss_mb": self._proc.memory_info().rss / _MIB,
        }

    def close(self) -> None:
        if self._nvml_up:
            self._nvml_up = False
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass


class SystemSampler(TrainingListener):
    """Emits a ``system`` event every ``interval_s`` seconds while the run is alive.

    ``sink`` is anything with ``emit(event_type, **fields)`` (the ``JsonlEventSink``).
    ``interval_s <= 0`` disables the sampler. Register it *before* the sink so it
    stops before ``run_end`` is written. The thread starts at the first epoch, after the
    sink has written ``run_start``. ``gpu_uuids`` are the visible GPUs in ``cuda:N`` order
    (:func:`visible_gpu_uuids`); ``probe_factory`` (no arguments) is for tests.
    """

    def __init__(self, sink, interval_s: float = 5.0, probe_factory=None,
                 gpu_uuids: list[str] | None = None):
        self._sink = sink
        self.interval_s = interval_s
        self._probe_factory = probe_factory or (lambda: SystemProbe(gpu_uuids=gpu_uuids))
        self._probe: SystemProbe | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False
        self._started = False

    @property
    def enabled(self) -> bool:
        return self.interval_s is not None and self.interval_s > 0

    def _warn_once(self, what: str, exc: BaseException) -> None:
        if not self._warned:
            self._warned = True
            warnings.warn(f"system sampler: {what} ({exc!r}); further errors are silenced",
                          RuntimeWarning, stacklevel=2)

    def _run(self, probe: SystemProbe) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                sample = probe.sample()
                if self._stop.is_set():     # stop() won while sampling: nothing after run_end
                    break
                self._sink.emit("system", **sample)
            except Exception as exc:  # noqa: BLE001 — sampling must never stop the run
                self._warn_once("sample not written", exc)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        try:
            self._probe = self._probe_factory()
        except Exception as exc:  # noqa: BLE001
            self._warn_once("disabled, probe failed", exc)
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(self._probe,), name="system-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5)
        if self._probe is not None and not (thread is not None and thread.is_alive()):
            self._probe.close()
        self._probe = None

    # ── TrainingListener ─────────────────────────────────────────────────────

    def on_epoch_start(self, epoch, epochs):
        # Not on_fit_start: the sink writes run_start there, after this listener.
        if not self._started:
            self._started = True
            self.start()

    def on_fit_end(self, best, epochs_run, early_stopped):
        self.stop()

    def on_fit_error(self, error, epochs_run, best):
        self.stop()
