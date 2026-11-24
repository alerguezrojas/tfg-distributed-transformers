"""Estimate of a run's epoch time from its manifest (src/web/run_estimate.py)."""
from __future__ import annotations

import pytest

from src.performance_model import MODEL_TABLE
from src.web.run_estimate import (
    estimate_run, inputs_from_run, measured_epoch_time_s, describe_throughput,
)

_MODEL = "vit_tiny_patch16_224"
_HEAVY = "vit_large_patch16_224"
_PANDA = {
    "dataset": "panda", "classes": ["a", "b", "c", "d", "e", "f"], "world_size": 1,
    "n_train": 7433, "n_val": 1593,
    "config": {"model": {"img_size": 512}, "data": {"image_size": 512},
               "training": {"batch_size": 16, "precision": "amp"}},
    "env": {"devices": ["NVIDIA GeForce RTX 3060 Ti"]},
}


def _info(name: str):
    return {"panda": (3, 6), "bigearthnet": (3, 19)}.get(name)


def _inputs(manifest=_PANDA, mode="single", cfg=None, model=_MODEL, precision="amp", **kw):
    return inputs_from_run(model=model, dataset="bigearthnet", mode=mode, precision=precision,
                           manifest=manifest, config_line=cfg or {}, dataset_info=_info, **kw)


def test_inputs_come_from_the_manifest():
    inp = _inputs()
    assert (inp.dataset, inp.img_size, inp.in_chans, inp.num_classes) == ("panda", 512, 3, 6)
    assert (inp.n_train, inp.n_val, inp.batch_per_gpu, inp.n_gpus) == (7433, 1593, 16, 1)
    assert inp.gpu == "NVIDIA GeForce RTX 3060 Ti" and inp.gpu_source == "run manifest"
    assert inp.missing == []


def test_estimate_profiles_the_model_at_the_runs_resolution():
    seen = {}

    def resolve(name, in_chans, num_classes, img_size):
        seen.update(name=name, in_chans=in_chans, num_classes=num_classes, img_size=img_size)
        return MODEL_TABLE[_MODEL]

    est = estimate_run(_inputs(), [100.0, 60.0, 62.0, 61.0], resolve=resolve)
    assert seen == {"name": _MODEL, "in_chans": 3, "num_classes": 6, "img_size": 512}
    assert est.skipped == "" and est.predicted_s > 0
    assert est.measured_s == pytest.approx(61.0)           # median without the first epoch
    assert est.error_pct == pytest.approx((est.predicted_s - 61.0) / 61.0 * 100)
    assert est.throughput_model                            # the origin is reported


def test_a_larger_resolution_gives_a_longer_estimate():
    """Fails if the resolution is dropped: the same spec at 224 px would be used."""
    base = {"vit": MODEL_TABLE[_HEAVY]}

    def resolve(name, in_chans, num_classes, img_size):
        from dataclasses import replace
        spec = base["vit"]
        return replace(spec, gflops_fwd=spec.gflops_fwd * (img_size / 224) ** 2,
                       calibrated=False)

    m224 = {**_PANDA, "config": {**_PANDA["config"], "model": {"img_size": 224}}}
    heavy = {"model": _HEAVY, "precision": "fp32"}      # compute-bound, so FLOPs show up
    t224 = estimate_run(_inputs(m224, **heavy), [1.0, 1.0], resolve=resolve).predicted_s
    t512 = estimate_run(_inputs(**heavy), [1.0, 1.0], resolve=resolve).predicted_s
    assert t512 > t224


def test_unknown_gpu_is_not_guessed():
    manifest = {**_PANDA, "env": {}}
    inp = _inputs(manifest)
    assert inp.gpu is None
    est = estimate_run(inp, [10.0, 10.0], resolve=lambda *a: MODEL_TABLE[_MODEL])
    assert est.predicted_s is None and "GPU name" in est.skipped
    assert est.measured_s == pytest.approx(10.0)


def test_gpu_fallback_is_labelled():
    manifest = {**_PANDA, "env": {}}
    inp = _inputs(manifest, fallback_gpu=("Tesla T4", "benchmark report of the same environment"))
    est = estimate_run(inp, [10.0, 10.0], resolve=lambda *a: MODEL_TABLE[_MODEL])
    assert est.gpu == "Tesla T4"
    assert any("benchmark report" in n for n in est.notes)


def test_gpu_outside_the_table_is_reported():
    manifest = {**_PANDA, "env": {"devices": ["Imaginary GPU 9000"]}}
    est = estimate_run(_inputs(manifest), [10.0], resolve=lambda *a: MODEL_TABLE[_MODEL])
    assert "not in the predictor's table" in est.skipped


def test_log_line_fills_what_the_manifest_lacks():
    manifest = {"dataset": "bigearthnet", "env": {"devices": ["Tesla T4", "Tesla T4"]}}
    cfg = {"batch": "16/GPU (global=32)", "train": "5000", "val": "1500"}
    inp = _inputs(manifest, mode="ddp", cfg=cfg)
    assert (inp.n_train, inp.n_val, inp.batch_per_gpu, inp.n_gpus) == (5000, 1500, 16, 2)
    assert inp.img_size == 224 and any("assumed 224" in n for n in inp.notes)
    est = estimate_run(inp, [50.0, 40.0], resolve=lambda *a: MODEL_TABLE[_MODEL])
    assert est.skipped == "" and est.predicted_s > 0


def test_missing_sizes_skip_the_estimate_and_say_so():
    manifest = {"dataset": "panda", "env": {"devices": ["Tesla T4"]}}
    inp = _inputs(manifest)
    est = estimate_run(inp, [10.0, 10.0], resolve=lambda *a: MODEL_TABLE[_MODEL])
    assert est.predicted_s is None
    assert "dataset size" in est.skipped and "batch size" in est.skipped


def test_data_parallel_batch_is_global():
    manifest = {**_PANDA, "world_size": 2}
    heavy = {"model": _HEAVY, "precision": "fp32"}
    inp = _inputs(manifest, mode="ddp", **heavy)
    assert inp.n_gpus == 2
    est2 = estimate_run(inp, [1.0, 1.0], resolve=lambda *a: MODEL_TABLE[_HEAVY])
    est1 = estimate_run(_inputs(**heavy), [1.0, 1.0], resolve=lambda *a: MODEL_TABLE[_HEAVY])
    assert est2.predicted_s < est1.predicted_s             # two GPUs, same per-GPU batch


@pytest.mark.parametrize("times,expected", [
    ([100.0, 10.0, 12.0, 11.0], 11.0), ([5.0], 5.0), ([], None),
    ([None, 7.0, 9.0], 9.0), ([float("nan"), 4.0], 4.0)])
def test_measured_epoch_time(times, expected):
    got = measured_epoch_time_s(times)
    assert got == expected if expected is None else got == pytest.approx(expected)


def test_throughput_description():
    assert "three terms" in describe_throughput("three-term")
    assert "FLOPs only" in describe_throughput("mfu")
