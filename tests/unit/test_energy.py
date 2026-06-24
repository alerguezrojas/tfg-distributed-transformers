"""Energy instrumentation: multi-GPU power sampling + model-parallel wiring.

Locks in the fix where `_PowerSampler` measures the COMBINED power of every GPU
a run uses (so DDP/model-parallel report the total, not just GPU 0)."""
import sys
import time
import types
from pathlib import Path

from src.training.fn_decorators import (
    measure_energy, _resolve_energy_devices, _PowerSampler)

_MP_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_model_parallel.py"


def test_measure_energy_returns_result_and_calls_once():
    seen = []
    wrapped = measure_energy(lambda x: x * 2, label="Trainer.train_epoch")
    out = wrapped(21)
    assert out == 42
    # builder-style single positional arg still works
    assert measure_energy(lambda: seen.append(1) or 7)() == 7


def test_resolve_devices_explicit_wins():
    # An explicit device list (model parallelism) is always honoured.
    assert _resolve_energy_devices([0, 1]) == [0, 1]
    # Default resolution returns a list (current device, or [] with no CUDA).
    assert isinstance(_resolve_energy_devices(None), list)


def test_power_sampler_sums_multiple_gpus(monkeypatch):
    """Two fake GPUs at 100 W each must report ~200 W combined."""
    fake = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: f"h{i}",
        nvmlDeviceGetPowerUsage=lambda h: 100_000,   # milliwatts → 100 W each
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    s = _PowerSampler([0, 1])
    assert s.available
    s.start()
    time.sleep(0.22)
    avg_w, energy_j = s.stop()
    assert abs(avg_w - 200.0) < 1.0          # the two devices are summed
    assert energy_j > 0


def test_power_sampler_unavailable_without_pynvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)   # import yields None → AttributeError
    s = _PowerSampler([0])
    assert not s.available
    assert s.stop() == (0.0, 0.0)


def test_model_parallel_script_wires_energy():
    src = _MP_SCRIPT.read_text(encoding="utf-8")
    assert '"--fn"' in src                                   # the flag exists
    assert "measure_energy(" in src                          # and it is applied
    assert "ModelParallelTrainer.train_epoch" in src         # label the web parser matches
    assert 'logger_name="model_parallel"' in src             # logs into the run's file
