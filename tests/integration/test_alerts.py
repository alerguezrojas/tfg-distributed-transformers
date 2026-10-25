"""Alerts wired through the builder (docs/inspeccion.md): a forced NaN is reported, the
run ends as it would without the monitor, and alerts never change the training."""
import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import registry
from src.data.registry import register_dataset
from src.data.spec import DatasetSpec
from src.training.builder import TrainingSessionBuilder
from src.training.monitor import layer_group


@pytest.fixture
def toy():
    names = []

    def _make(name, nan_train=False):
        def build(data_cfg, split, transform):
            g = torch.Generator().manual_seed(0 if split == "train" else 1)
            x = torch.randn(24, 3, 32, 32, generator=g)
            if nan_train and split == "train":
                x[8:] = float("nan")
            y = torch.eye(3)[torch.randint(0, 3, (24,), generator=g)]
            return TensorDataset(x, y)
        spec = DatasetSpec(name=name, description="synthetic", classes=("a", "b", "c"),
                           task="multiclass", in_channels=3, required_keys=(), build=build,
                           transforms=lambda split: None)
        names.append(name)
        return register_dataset(spec)

    yield _make
    for name in names:
        registry._REGISTRY.pop(name, None)


def _train(tmp_path, monkeypatch, spec, monitor=None, layers=("confusion",), epochs=2,
           seed=7):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = {"data": {"name": spec.name, "num_workers": 0},
           "model": {"name": "resnet18", "pretrained": False},
           "training": {"epochs": epochs, "batch_size": 8, "lr": 0.001, "log_batch_every": 1,
                        "mixup_alpha": 0.2, "label_smoothing": 0.1},
           "checkpoint": {"dir": str(tmp_path / "ckpt")}, "output": {"env": "test"}}
    if monitor is not None:
        cfg["monitor"] = monitor
    trainer = (TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="t")
               .with_layers(*layers).build())
    dl_train = DataLoader(spec.dataset(cfg["data"], "train"), batch_size=8, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    dl_val = DataLoader(spec.dataset(cfg["data"], "val"), batch_size=8)
    trainer.fit(dl_train, dl_val, epochs=epochs)
    log_dir = tmp_path / "logs" / "test" / spec.name / "single" / "resnet18"
    events = [json.loads(x) for x in (log_dir / "events_t.jsonl").read_text().splitlines()]
    return trainer, events


def _alerts(events):
    return [e for e in events if e["type"] == "alert"]


def _lifecycle(events):
    return [e["type"] for e in events if e["type"] not in ("alert", "heartbeat", "system")]


def test_forced_nan_raises_a_critical_alert_and_the_run_is_unchanged(tmp_path, monkeypatch, toy):
    spec = toy("toy_nan", nan_train=True)
    layers = ("confusion", "batch-monitor")
    _, with_alerts = _train(tmp_path / "on", monkeypatch, spec, layers=layers)
    _, without = _train(tmp_path / "off", monkeypatch, spec, layers=layers,
                        monitor={"alerts": {"enabled": False}})
    nan = [a for a in _alerts(with_alerts)
           if a["code"] == "nan_loss" and a["group"] == "train"]
    assert len(nan) == 1                                    # once, not per batch / epoch
    assert nan[0]["level"] == "critical" and nan[0]["epoch"] == 1
    assert nan[0]["batch"] is not None                      # batch-monitor: found in the batch
    assert nan[0]["value"] is None                          # NaN is not JSON
    assert set(nan[0]) >= {"type", "level", "code", "message", "epoch", "batch", "value",
                           "threshold", "group"}
    assert not _alerts(without)
    assert with_alerts[-1]["status"] == without[-1]["status"]
    assert _lifecycle(with_alerts) == _lifecycle(without)


def test_without_batch_hooks_the_epoch_loss_is_checked(tmp_path, monkeypatch, toy):
    spec = toy("toy_nan_epoch", nan_train=True)
    _, events = _train(tmp_path, monkeypatch, spec)         # no batch-monitor, no layers
    (a,) = [a for a in _alerts(events)
            if a["code"] == "nan_loss" and a["group"] == "train"]
    assert a["batch"] is None and a["epoch"] == 1 and a["level"] == "critical"
    kinds = [e["type"] for e in events]
    assert "batch" not in kinds                             # nothing forced per batch
    assert kinds.index("alert") > kinds.index("epoch_end")


def test_alerts_do_not_change_the_training(tmp_path, monkeypatch, toy):
    spec = toy("toy_alert_inv")
    mon = {"enabled": True, "samples": 8, "gradcam": 0}
    off, _ = _train(tmp_path / "off", monkeypatch, spec,
                    monitor={**mon, "alerts": {"enabled": False}})
    rng_off, py_off = torch.get_rng_state(), random.getstate()
    # Thresholds that fire on this tiny run, so the alert paths really execute.
    on, events = _train(tmp_path / "on", monkeypatch, spec,
                        monitor={**mon, "alerts": {"update_ratio_min": 1.0,
                                                   "update_ratio_patience": 1,
                                                   "data_wait_frac": 0.0}})
    assert _alerts(events)
    for (n1, p1), (n2, p2) in zip(off.model.named_parameters(), on.model.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), f"{n1} differs with alerts on"
    assert torch.equal(rng_off, torch.get_rng_state()) and random.getstate() == py_off


def test_healthy_run_has_no_alerts(tmp_path, monkeypatch, toy):
    _, events = _train(tmp_path, monkeypatch, toy("toy_quiet"))
    assert not _alerts(events)


def test_gradient_histograms_are_emitted_next_to_the_weight_ones(tmp_path, monkeypatch, toy):
    spec = toy("toy_gradhist")
    trainer, events = _train(tmp_path, monkeypatch, spec,
                             monitor={"enabled": True, "samples": 8, "gradcam": 0})
    hists = [e for e in events if e["type"] == "histograms"]
    assert [(h["epoch"], h["kind"]) for h in hists] == [
        (1, "weights"), (1, "gradients"), (2, "weights"), (2, "gradients")]
    for g in hists[2]["groups"] + hists[3]["groups"]:
        assert len(g["edges"]) == 33 and len(g["counts"]) == 32
    grads = {g["name"]: g for g in hists[3]["groups"]}
    model = trainer
    while not hasattr(model, "model"):
        model = model._trainer
    sizes: dict[str, int] = {}
    for name, p in model.model.named_parameters():
        if p.grad is not None:
            sizes[layer_group(name)] = sizes.get(layer_group(name), 0) + p.numel()
    # Counts cover every gradient element of the last batch, and match the live .grad.
    assert {n: sum(g["counts"]) for n, g in grads.items()} == sizes
    name = next(iter(sizes))
    gmax = max(p.grad.max().item() for n, p in model.model.named_parameters()
               if layer_group(n) == name and p.grad is not None)
    assert grads[name]["edges"][-1] == pytest.approx(gmax, rel=1e-5)
