"""Estimate vs Benchmark vs Run: the record of a run uses its own setup (manifest)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.web.tabs.benchmark.validate import _run_record


def _panda_run(tmp_path, with_gpu=True):
    """A PANDA run at 512 px with a manifest (and, optionally, the GPU it ran on)."""
    log = tmp_path / "train_panda.log"
    log.write_text(
        "2026-11-02 [INFO] Configuración: modelo=vit_tiny_patch16_224 | batch=16/GPU (global=16) "
        "| epochs=3 | precision=amp | train=7433 | val=1593 | dataset=panda\n"
        "ETA: 00:00 (90.0s/epoch)\n")
    manifest = {"v": 1, "type": "run_manifest", "dataset": "panda", "classes": list("abcdef"),
                "mode": "single", "world_size": 1,
                "config": {"model": {"img_size": 512}, "training": {"batch_size": 16}},
                "env": {"devices": ["NVIDIA GeForce RTX 3060 Ti"] if with_gpu else []}}
    mpath = tmp_path / "run_x.json"
    mpath.write_text(json.dumps(manifest))
    return SimpleNamespace(log_path=log, epoch_csv_path=None, precision="amp", mode="single",
                           model="vit_tiny_patch16_224", env="local", dataset="panda",
                           manifest_path=mpath, label="02/11/2026 vit_tiny")


def test_run_record_has_an_estimate_and_no_f1_prior_outside_bigearthnet(tmp_path):
    rec, _ = _run_record(_panda_run(tmp_path), None, None)
    est = rec["est"]
    assert est.skipped == "" and est.predicted_s > 0
    assert rec["t"]["a"] == pytest.approx(est.predicted_s / 60.0)
    assert rec["f"] == {"a": None, "b": None, "r": None}     # no prior for PANDA
    assert "no quality prior outside BigEarthNet" in rec["note"]


def test_run_record_does_not_guess_the_gpu(tmp_path):
    rec, _ = _run_record(_panda_run(tmp_path, with_gpu=False), None, None)
    assert rec["est"].predicted_s is None and "GPU name" in rec["est"].skipped
    assert rec["t"]["a"] is None
