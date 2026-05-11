"""Python @ function decorators for cross-cutting utility concerns.

These wrap individual functions (not objects), making them composable with
the standard @ syntax or by direct application at runtime:

    trainer.train_epoch = measure_energy(timed(trainer.train_epoch))

Available decorators
--------------------
timed             — prints wall-clock execution time
log_call          — prints when a function starts and finishes
retry_on_cuda_oom — retries once after clearing the CUDA cache on OOM
measure_energy    — samples GPU power consumption and reports Joules / Wh
"""

import functools
import threading
import time


# ── timed ────────────────────────────────────────────────────────────────────

def timed(fn):
    """Print the wall-clock execution time of any function."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        print(f"[timed] {fn.__qualname__}: {time.time() - t0:.2f}s")
        return result
    return wrapper


# ── log_call ─────────────────────────────────────────────────────────────────

def log_call(fn):
    """Print when a function is called and when it returns."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[call] → {fn.__qualname__}")
        result = fn(*args, **kwargs)
        print(f"[call] ← {fn.__qualname__} done")
        return result
    return wrapper


# ── retry_on_cuda_oom ─────────────────────────────────────────────────────────

def retry_on_cuda_oom(fn):
    """Retry the function once after clearing the CUDA cache on OOM."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import torch
        try:
            return fn(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[oom] CUDA OOM en {fn.__qualname__} — liberando caché y reintentando")
            return fn(*args, **kwargs)
    return wrapper


# ── measure_energy ────────────────────────────────────────────────────────────

class _PowerSampler:
    """Background thread that reads GPU power via pynvml every 100 ms."""

    def __init__(self, device_index: int = 0):
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._handle is not None

    def start(self):
        if not self.available:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import pynvml
        while not self._stop.is_set():
            try:
                self._samples.append(pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0)
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self) -> tuple[float, float]:
        """Return (avg_watts, energy_joules)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self._samples:
            return 0.0, 0.0
        avg_w = sum(self._samples) / len(self._samples)
        return avg_w, avg_w * len(self._samples) * 0.1


def measure_energy(fn):
    """Sample GPU power consumption during execution and report Joules / Wh.

    Requires nvidia-ml-py (installed with the project).
    Falls back silently if no GPU is present or pynvml is unavailable.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        sampler = _PowerSampler()
        sampler.start()
        result = fn(*args, **kwargs)
        avg_w, energy_j = sampler.stop()
        if sampler.available:
            print(
                f"[energy] {fn.__qualname__}: "
                f"{energy_j:.1f} J  ({energy_j / 3600:.5f} Wh)  "
                f"potencia media {avg_w:.1f} W"
            )
        else:
            print(f"[energy] {fn.__qualname__}: GPU no disponible — medición omitida")
        return result
    return wrapper
