import functools
import threading
import time


class _PowerSampler:
    """Background thread that samples GPU power via pynvml every 100 ms."""

    def __init__(self, device_index: int = 0):
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._handle = None
        self._thread: threading.Thread | None = None
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
                mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples.append(mw / 1000.0)  # convert mW → W
            except Exception:
                pass
            time.sleep(0.1)

    def stop(self) -> tuple[float, float]:
        """Stop sampling. Returns (avg_watts, energy_joules)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self._samples:
            return 0.0, 0.0
        avg_w = sum(self._samples) / len(self._samples)
        energy_j = avg_w * len(self._samples) * 0.1  # W × samples × interval
        return avg_w, energy_j


def measure_energy(fn):
    """Measure GPU energy consumption (Joules and Watts) during execution.

    Requires pynvml: uv add pynvml
    Falls back silently if pynvml is not installed or no GPU is present.

    Adds 'energy_j' and 'power_w' to the result dict when measurement is available.

    Example:
        trainer.train_epoch = measure_energy(trainer.train_epoch)
        # or combined:
        trainer.train_epoch = measure_energy(timed(trainer.train_epoch))
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        sampler = _PowerSampler()
        sampler.start()

        result = fn(*args, **kwargs)

        avg_w, energy_j = sampler.stop()

        if sampler.available:
            energy_wh = energy_j / 3600.0
            print(
                f"[energy] {fn.__qualname__}: "
                f"{energy_j:.1f} J  ({energy_wh:.5f} Wh)  "
                f"potencia media {avg_w:.1f} W"
            )
            if isinstance(result, dict):
                result = {**result, "energy_j": energy_j, "power_w": avg_w}
        else:
            print(f"[energy] {fn.__qualname__}: pynvml no disponible — medición omitida")

        return result
    return wrapper
