"""SystemSampler / SystemProbe: eventos `system` con NVML y psutil simulados.

Contrato de los campos en docs/inspeccion.md. Nada de esto toca la GPU real salvo la
última prueba, que se salta si NVML no ve ninguna.
"""
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from src.training.events import JsonlEventSink
from src.training.system_sampler import SystemProbe, SystemSampler

GPU_FIELDS = {"index", "name", "util_pct", "mem_used_mb", "mem_total_mb", "power_w",
              "temp_c", "sm_clock_mhz"}
EVENT_FIELDS = {"gpus", "cpu_pct", "ram_used_pct", "proc_cpu_pct", "proc_rss_mb"}


class FakeNvml:
    """Dos GPUs; el handle es el propio índice físico."""
    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1

    def __init__(self, n=2, unsupported=(), fail_init=False):
        self.n, self.unsupported, self.fail_init = n, set(unsupported), fail_init
        self.inits = self.shutdowns = 0
        self.uuids = {f"GPU-uuid-{i}": i for i in range(n)}

    def _check(self, name):
        if name in self.unsupported:
            raise RuntimeError(f"{name}: not supported")

    def nvmlInit(self):
        if self.fail_init:
            raise RuntimeError("driver not loaded")
        self.inits += 1

    def nvmlShutdown(self):
        self.shutdowns += 1

    def nvmlDeviceGetCount(self):
        return self.n

    def nvmlDeviceGetHandleByIndex(self, i):
        if not 0 <= i < self.n:
            raise RuntimeError("invalid index")
        return i

    def nvmlDeviceGetHandleByUUID(self, uuid):
        return self.uuids[uuid]

    def nvmlDeviceGetName(self, h):
        return f"GPU-{h}".encode()                      # bytes, como las versiones viejas

    def nvmlDeviceGetUtilizationRates(self, h):
        self._check("util")
        return SimpleNamespace(gpu=40 + h, memory=10)

    def nvmlDeviceGetMemoryInfo(self, h):
        self._check("mem")
        return SimpleNamespace(used=(100 + h) * 1024 ** 2, total=8192 * 1024 ** 2)

    def nvmlDeviceGetPowerUsage(self, h):
        self._check("power")
        return 75_500 + h                               # mW

    def nvmlDeviceGetTemperature(self, h, kind):
        return 60 + h

    def nvmlDeviceGetClockInfo(self, h, kind):
        return 1700 + h


class FakeProcess:
    def cpu_percent(self, interval=None):
        return 123.0

    def memory_info(self):
        return SimpleNamespace(rss=512 * 1024 ** 2)


class FakePsutil:
    Process = FakeProcess

    @staticmethod
    def cpu_percent(interval=None):
        return 12.5

    @staticmethod
    def virtual_memory():
        return SimpleNamespace(percent=33.0)


def _probe(nvml=None, visible=None):
    return SystemProbe(nvml=nvml or FakeNvml(), psutil_mod=FakePsutil(),
                       visible_devices=visible)


# ── SystemProbe ───────────────────────────────────────────────────────────────

def test_sample_has_exactly_the_contract_fields():
    s = _probe().sample()
    assert set(s) == EVENT_FIELDS
    assert len(s["gpus"]) == 2
    assert all(set(g) == GPU_FIELDS for g in s["gpus"])
    json.dumps(s, allow_nan=False)                      # serializable tal cual


def test_sample_values_and_units():
    s = _probe().sample()
    g1 = s["gpus"][1]
    assert g1 == {"index": 1, "name": "GPU-1", "util_pct": 41, "mem_used_mb": 101.0,
                  "mem_total_mb": 8192.0, "power_w": pytest.approx(75.501),
                  "temp_c": 61, "sm_clock_mhz": 1701}
    assert (s["cpu_pct"], s["ram_used_pct"], s["proc_cpu_pct"], s["proc_rss_mb"]) == (
        12.5, 33.0, 123.0, 512.0)


def test_unsupported_query_gives_null_not_failure():
    s = _probe(FakeNvml(unsupported={"power", "mem"})).sample()
    g = s["gpus"][0]
    assert g["power_w"] is None and g["mem_used_mb"] is None and g["mem_total_mb"] is None
    assert g["util_pct"] == 40 and g["temp_c"] == 60


def test_no_nvml_driver_means_no_gpus():
    nvml = FakeNvml(fail_init=True)
    s = _probe(nvml).sample()
    assert s["gpus"] == [] and set(s) == EVENT_FIELDS


def test_pynvml_not_installed_means_no_gpus(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)    # `import pynvml` → ImportError
    s = SystemProbe(psutil_mod=FakePsutil(), visible_devices=None).sample()
    assert s["gpus"] == []


def test_zero_gpus():
    assert _probe(FakeNvml(n=0)).sample()["gpus"] == []


@pytest.mark.parametrize("visible, expected_names", [
    (None, ["GPU-0", "GPU-1"]),                         # sin la variable: todas
    ("1", ["GPU-1"]),
    ("1,0", ["GPU-1", "GPU-0"]),
    ("", []),                                           # vacía: ninguna (como CUDA)
    ("GPU-uuid-1", ["GPU-1"]),
    ("0,7,1", ["GPU-0"]),                               # CUDA para en el primer inválido
])
def test_cuda_visible_devices_is_respected(visible, expected_names):
    gpus = _probe(visible=visible).sample()["gpus"]
    assert [g["name"] for g in gpus] == expected_names
    assert [g["index"] for g in gpus] == list(range(len(expected_names)))   # ordinal visible


def test_cuda_visible_devices_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    gpus = SystemProbe(nvml=FakeNvml(), psutil_mod=FakePsutil()).sample()["gpus"]
    assert [g["name"] for g in gpus] == ["GPU-1"]


def test_close_shuts_nvml_down_once():
    nvml = FakeNvml()
    probe = _probe(nvml)
    probe.close()
    probe.close()
    assert (nvml.inits, nvml.shutdowns) == (1, 1)


# ── SystemSampler ─────────────────────────────────────────────────────────────

class ListSink:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def emit(self, event_type, **fields):
        with self.lock:
            self.events.append((event_type, fields))


def _factory(nvml=None):
    return lambda: SystemProbe(nvml=nvml or FakeNvml(), psutil_mod=FakePsutil(),
                               visible_devices=None)


def _wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_emits_system_events_at_the_interval_and_stops():
    sink = ListSink()
    sampler = SystemSampler(sink, interval_s=0.02, probe_factory=_factory())
    sampler.on_fit_start(3)
    assert _wait_for(lambda: len(sink.events) >= 3)
    sampler.on_fit_end(0.9, 3, False)
    n = len(sink.events)
    time.sleep(0.1)
    assert len(sink.events) == n                         # parado: no emite más
    assert sampler._thread is None
    assert all(t == "system" and set(f) == EVENT_FIELDS for t, f in sink.events)


def test_first_sample_waits_one_interval():
    sink = ListSink()
    sampler = SystemSampler(sink, interval_s=60, probe_factory=_factory())
    sampler.on_fit_start(1)
    time.sleep(0.05)
    assert sink.events == []                             # nada antes de run_start
    t0 = time.monotonic()
    sampler.on_fit_end(0.0, 0, False)
    assert time.monotonic() - t0 < 2                     # se para sin esperar el intervalo


def test_stops_on_fit_error_too():
    sink = ListSink()
    sampler = SystemSampler(sink, interval_s=0.01, probe_factory=_factory())
    sampler.on_fit_start(1)
    thread = sampler._thread
    sampler.on_fit_error(RuntimeError("boom"), 0, -1.0)
    assert not thread.is_alive()


def test_nvml_is_shut_down_after_the_run():
    nvml = FakeNvml()
    sampler = SystemSampler(ListSink(), interval_s=0.01, probe_factory=_factory(nvml))
    sampler.on_fit_start(1)
    sampler.on_fit_end(0.0, 0, False)
    assert (nvml.inits, nvml.shutdowns) == (1, 1)


@pytest.mark.parametrize("interval", [0, -1, None])
def test_disabled_interval_starts_nothing(interval):
    sink = ListSink()
    sampler = SystemSampler(sink, interval_s=interval, probe_factory=_factory())
    sampler.on_fit_start(1)
    assert sampler._thread is None and not sampler.enabled
    sampler.on_fit_end(0.0, 0, False)                    # no falla
    assert sink.events == []


def test_write_errors_warn_once_and_never_propagate():
    class BrokenSink:
        calls = 0

        def emit(self, *a, **k):
            BrokenSink.calls += 1
            raise OSError("disk full")

    sampler = SystemSampler(BrokenSink(), interval_s=0.01, probe_factory=_factory())
    with pytest.warns(RuntimeWarning, match="sample not written") as record:
        sampler.on_fit_start(1)
        assert _wait_for(lambda: BrokenSink.calls >= 3)  # el hilo sigue vivo tras fallar
        sampler.on_fit_end(0.0, 0, False)
    assert len([w for w in record if "system sampler" in str(w.message)]) == 1


def test_sampling_errors_warn_once_and_never_propagate():
    class BadProbe:
        n = 0

        def sample(self):
            BadProbe.n += 1
            raise RuntimeError("psutil exploded")

        def close(self):
            pass

    sink = ListSink()
    sampler = SystemSampler(sink, interval_s=0.01, probe_factory=BadProbe)
    with pytest.warns(RuntimeWarning, match="sample not written"):
        sampler.on_fit_start(1)
        assert _wait_for(lambda: BadProbe.n >= 3)
        sampler.on_fit_end(0.0, 0, False)
    assert sink.events == []


def test_probe_failure_disables_the_sampler_without_raising():
    def broken():
        raise RuntimeError("no psutil")

    sampler = SystemSampler(ListSink(), interval_s=0.01, probe_factory=broken)
    with pytest.warns(RuntimeWarning, match="probe failed"):
        sampler.on_fit_start(1)
    assert sampler._thread is None
    sampler.on_fit_end(0.0, 0, False)


def test_events_reach_the_jsonl_stream(tmp_path):
    path = tmp_path / "events_t.jsonl"
    sink = JsonlEventSink(path, run_info={"run_id": "t"}, heartbeat_s=0)
    sampler = SystemSampler(sink, interval_s=0.01, probe_factory=_factory())
    sampler.on_fit_start(1)
    sink.on_fit_start(1)
    assert _wait_for(lambda: path.read_text().count('"system"') >= 2)
    sampler.on_fit_end(0.0, 0, False)
    sink.on_fit_end(0.0, 0, False)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    system = [e for e in lines if e["type"] == "system"]
    assert system and all(e["v"] == 1 and set(e) - {"v", "type", "time"} == EVENT_FIELDS
                          for e in system)
    assert lines[-1]["type"] == "run_end"                # nada después del run_end


# ── NVML real (solo con GPU) ──────────────────────────────────────────────────

def _real_gpu():
    try:
        import pynvml
        pynvml.nvmlInit()
        ok = pynvml.nvmlDeviceGetCount() > 0
        pynvml.nvmlShutdown()
        return ok
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _real_gpu(), reason="NVML no ve ninguna GPU")
def test_real_nvml_sample():
    probe = SystemProbe()
    try:
        s = probe.sample()
    finally:
        probe.close()
    assert s["gpus"] and set(s["gpus"][0]) == GPU_FIELDS
    g = s["gpus"][0]
    assert g["mem_total_mb"] > 0 and 0 <= g["mem_used_mb"] <= g["mem_total_mb"]
    assert g["name"]
    json.dumps(s, allow_nan=False)
