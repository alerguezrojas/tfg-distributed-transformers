"""Synthetic run for the Inspector page: ``events_<ts>.jsonl`` + ``inspect_<ts>/``.

Follows docs/inspeccion.md: besides the v1 events it writes ``monitor_start``, ``system``,
``layers``, ``histograms`` and ``inspection``, the new ``batch`` / ``epoch_end.train`` fields
(``data_time_s``, ``step_time_s``, ``samples_per_s``, ``data_wait_frac``) and the inspection
directory made by ``make_inspect_fixture``. Nothing is trained.

    python tests/fixtures/make_inspector_run.py ROOT [multilabel|binary|multiclass]
"""
import json
import sys
from pathlib import Path

import numpy as np

from tests.fixtures.make_inspect_fixture import make_inspect_fixture

T0 = 1792231200.0
GROUPS = ["backbone.patch_embed", "backbone.blocks.0", "backbone.blocks.1", "head.1"]


def _w(f, **ev):
    f.write(json.dumps({"v": 1, **ev}) + "\n")


def make_inspector_run(root, task: str = "multilabel", ts: str = "25102026_120000",
                       epochs: int = 3, c: int = 5, m: int = 48, n_batches: int = 20,
                       dataset: str = "medmnist_chest", model: str = "vit_tiny_patch16_224",
                       with_inspect: bool = True, with_events: bool = True,
                       finished: bool = True, seed: int = 0) -> Path:
    """Create the run under ``<root>/logs/local/<dataset>/single/<model>/``; return that folder."""
    rng = np.random.default_rng(seed)
    folder = Path(root) / "logs" / "local" / dataset / "single" / model
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"train_{ts}.log").write_text("x\n", encoding="utf-8")
    c = 1 if task == "binary" else c
    if with_inspect:
        make_inspect_fixture(folder, ts=ts, task=task, c=c, m=m, epochs=tuple(range(1, epochs + 1)),
                             seed=seed)
    if not with_events:
        return folder
    classes = [f"class_{i}" for i in range(c)]
    t = T0
    with (folder / f"events_{ts}.jsonl").open("w", encoding="utf-8") as f:
        _w(f, type="run_start", time=t, run_id=ts, dataset=dataset, task=task, classes=classes,
           model=model, mode="single", world_size=1, epochs=epochs, select_metric="auc",
           config={"model": {"name": model}, "training": {"epochs": epochs}},
           env={"host": "pc", "python": "3.12", "torch": "2.5", "timm": "1.0",
                "git_commit": None, "devices": ["NVIDIA GeForce RTX 3060 Ti"]})
        if with_inspect:
            _w(f, type="monitor_start", time=t, dir=f"inspect_{ts}", samples=m, panel=4,
               classes=classes, full_val=True, embedding_dim=16)
        for ep in range(1, epochs + 1):
            _w(f, type="epoch_start", time=t, epoch=ep, epochs=epochs)
            for b in range(5, n_batches + 1, 5):
                t += 2.0
                data_t = float(rng.uniform(0.01, 0.05))
                step_t = float(rng.uniform(0.1, 0.2))
                _w(f, type="batch", time=t, epoch=ep, batch=b, n_batches=n_batches,
                   metrics={"batch_loss": 0.5 / ep, "running_loss": 0.5 / ep, "lr": 1e-4,
                            "data_time_s": data_t, "step_time_s": step_t,
                            "samples_per_s": 32 / (data_t + step_t)})
                _w(f, type="system", time=t, cpu_pct=30.0 + ep, ram_used_pct=40.0,
                   proc_cpu_pct=110.0, proc_rss_mb=2500.0,
                   gpus=[{"index": g, "name": "NVIDIA GeForce RTX 3060 Ti",
                          "util_pct": float(rng.uniform(60, 95)), "mem_used_mb": 3000.0 + 100 * ep,
                          "mem_total_mb": 8192.0, "power_w": float(rng.uniform(100, 160)),
                          "temp_c": float(rng.uniform(55, 70)), "sm_clock_mhz": 1700.0}
                         for g in range(2)])
                _w(f, type="layers", time=t, epoch=ep, batch=b, groups=[
                    {"name": g, "grad_norm": float(rng.uniform(0.1, 2.0)) / ep,
                     "weight_norm": 10.0 + i, "update_ratio": float(10 ** rng.uniform(-5, -1.5))}
                    for i, g in enumerate(GROUPS)])
            t += 1.0
            _w(f, type="histograms", time=t, epoch=ep, groups=[
                {"name": g, "edges": np.linspace(-0.5, 0.5, 33).round(4).tolist(),
                 "counts": rng.integers(0, 500, 32).tolist()} for g in GROUPS])
            _w(f, type="epoch_end", time=t, epoch=ep, epochs=epochs,
               train={"loss": 0.5 / ep, "f1": 0.3 + 0.1 * ep, "samples_per_s": 180.0 + ep,
                      "data_wait_frac": 0.15 / ep},
               val={"loss": 0.4 / ep, "f1": 0.4, "auc": 0.6 + 0.1 * ep, "accuracy": 0.7},
               epoch_time_s=30.0, best=0.6 + 0.1 * ep, is_best=True)
            if with_inspect:
                _w(f, type="inspection", time=t, epoch=ep, file=f"inspect_{ts}/epoch_{ep:03d}.npz",
                   seconds=0.4)
        if finished:
            _w(f, type="run_end", time=t + 1, status="completed", epochs_run=epochs,
               best=0.6 + 0.1 * epochs, total_time_s=t + 1 - T0)
    return folder


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(make_inspector_run(sys.argv[1], task=sys.argv[2] if len(sys.argv) > 2 else "multilabel"))
