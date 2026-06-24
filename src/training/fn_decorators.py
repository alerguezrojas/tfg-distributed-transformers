"""Python @ function decorators for cross-cutting utility concerns.

These wrap individual functions (not objects), making them composable with
the standard @ syntax or by direct application at runtime:

    trainer.train_epoch = measure_energy(timed(trainer.train_epoch))

Available decorators
--------------------
timed             — logs wall-clock execution time
log_call          — logs when a function starts and finishes
retry_on_cuda_oom — retries once after clearing the CUDA cache on OOM
measure_energy    — samples GPU power consumption and reports Joules / Wh
"""

import functools
import logging
import threading
import time


def _log(msg: str, logger_name: str = "trainer") -> None:
    """Route to the given logger if it has handlers, otherwise print to stdout."""
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        logger.info(msg)
    else:
        print(msg)


# ── timed ────────────────────────────────────────────────────────────────────

def timed(fn):
    """Log the wall-clock execution time of any function."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        _log(f"[timed] {fn.__qualname__}: {time.time() - t0:.2f}s")
        return result
    return wrapper


# ── log_call ─────────────────────────────────────────────────────────────────

def log_call(fn):
    """Log when a function is called and when it returns."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _log(f"[call] → {fn.__qualname__}")
        result = fn(*args, **kwargs)
        _log(f"[call] ← {fn.__qualname__} done")
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
            _log(f"[oom] CUDA OOM en {fn.__qualname__} — liberando caché y reintentando")
            return fn(*args, **kwargs)
    return wrapper


# ── measure_energy ────────────────────────────────────────────────────────────

class _PowerSampler:
    """Background thread that reads GPU power via pynvml every 100 ms.

    Samples one OR several GPUs and reports their *combined* power, so a run
    that spans more than one device (DDP, model parallelism) gets the total
    energy of all the GPUs it uses, not just the first one.
    """

    def __init__(self, device_indices=(0,)):
        if isinstance(device_indices, int):
            device_indices = (device_indices,)
        self._samples: list[float] = []       # each sample = SUM of the devices' watts
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handles: list = []
        self._error: str | None = None
        try:
            import pynvml
            pynvml.nvmlInit()
            for i in device_indices:
                self._handles.append(pynvml.nvmlDeviceGetHandleByIndex(int(i)))
        except ImportError:
            self._error = "pynvml no instalado"
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        return len(self._handles) > 0

    def start(self):
        if not self.available:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import pynvml
        while not self._stop.is_set():
            try:
                total_w = sum(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0 for h in self._handles)
                self._samples.append(total_w)
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self) -> tuple[float, float]:
        """Return (avg_watts, energy_joules) summed across the sampled devices."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self._samples:
            return 0.0, 0.0
        avg_w = sum(self._samples) / len(self._samples)
        return avg_w, avg_w * len(self._samples) * 0.1


def _resolve_energy_devices(explicit):
    """Physical GPU indices this process should sample (or None to skip logging).

    - ``explicit`` (e.g. model parallelism passes its split devices) wins.
    - In DDP, only rank 0 logs: it samples ALL the run's GPUs on a single node
      (so the figure is the total across GPUs), or just its own across nodes.
    - Otherwise (single process) it samples the current device.
    """
    try:
        import torch
    except Exception:
        return list(explicit) if explicit is not None else []
    if not torch.cuda.is_available():
        return list(explicit) if explicit is not None else []
    if explicit is not None:
        return [int(d) for d in explicit]
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return None                      # other ranks don't measure/log
            world = dist.get_world_size()
            ndev = torch.cuda.device_count()
            if world <= ndev:                    # single node: sum all the run's GPUs
                return list(range(world))
            return [torch.cuda.current_device()]  # multi-node: only own GPU is local
    except Exception:
        pass
    return [torch.cuda.current_device()]


def measure_energy(fn, devices=None, label: str | None = None,
                   logger_name: str = "trainer"):
    """Sample GPU power consumption during execution and report Joules / Wh.

    Reports the energy of ALL the GPUs the run uses (see ``_resolve_energy_devices``):
    the current device for a single-GPU run, every GPU on the node for DDP (logged
    once, by rank 0), or an explicit ``devices`` list for model parallelism.

    ``label`` overrides the line's name (e.g. ``ModelParallelTrainer.train_epoch``
    so the web parser recognises it); ``logger_name`` routes to the right logger.
    Requires nvidia-ml-py; falls back silently if no GPU / pynvml is present.
    """
    name = label or getattr(fn, "__qualname__", "fn")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        dev = _resolve_energy_devices(devices)
        if dev is None:                       # this rank does not measure/log
            return fn(*args, **kwargs)
        sampler = _PowerSampler(dev)
        sampler.start()
        result = fn(*args, **kwargs)
        avg_w, energy_j = sampler.stop()
        if sampler.available:
            _log(
                f"[energy] {name}: "
                f"{energy_j:.1f} J  ({energy_j / 3600:.5f} Wh)  "
                f"potencia media {avg_w:.1f} W",
                logger_name,
            )
        else:
            logging.getLogger(logger_name).debug(
                f"[energy] {name}: GPU no disponible ({sampler._error})")
        return result
    return wrapper
