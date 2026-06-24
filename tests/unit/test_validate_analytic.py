"""Benchmark-vs-Run: model-parallel / heterogeneous runs get an analytic speedup.

The empirical benchmark only extrapolates data-parallel (DDP) scaling, so without
this analytic fallback an MP run would show no estimate in the *Benchmark vs Run* tab.
"""
import itertools
from types import SimpleNamespace

from src.web.tabs.benchmark.validate import _analytic_speedup

_counter = itertools.count()


def _fake_run(tmp_path, mode, batch="96 (global)", model="vit_base_patch16_224"):
    # Unique filename per call: _run_config caches by path, so reusing one file
    # would return a stale config after overwriting it.
    log = tmp_path / f"train_{next(_counter)}.log"
    log.write_text(
        "2026-06-24 18:59:49 [INFO ] Configuración: "
        f"modelo={model} | batch={batch} | epochs=15 | precision=fp32 | "
        "train=5000 | val=1500\n"
    )
    return SimpleNamespace(mode=mode, model=model, precision="fp32", log_path=log)


def test_model_parallel_gets_analytic_speedup_near_one(tmp_path):
    r = _fake_run(tmp_path, "model_parallel")
    sp = _analytic_speedup(r, {"hardware_name": "Tesla T4"})
    assert sp is not None
    assert 0.85 <= sp <= 1.15          # naive pipeline does not accelerate (≈1×)


def test_analytic_speedup_needs_gpu_and_batch(tmp_path):
    # No GPU recorded in the benchmark report → cannot predict.
    assert _analytic_speedup(_fake_run(tmp_path, "model_parallel"), {}) is None
    # No batch in the run config → cannot predict.
    r = _fake_run(tmp_path, "model_parallel", batch="")
    assert _analytic_speedup(r, {"hardware_name": "Tesla T4"}) is None
