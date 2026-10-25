"""Prediction and layer monitors (docs/inspeccion.md): contract of the saved data,
Grad-CAM, and — most important — that monitoring never changes the training."""
import json
import random
import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import registry
from src.data.registry import register_dataset
from src.data.spec import DatasetSpec
from src.training.builder import TrainingSessionBuilder
from src.training.monitor import (
    grad_cam, layer_group, panel_positions, select_indices, thumbnails,
)


class _Labelled(TensorDataset):
    @property
    def targets(self):
        return self.tensors[1]


def _build(n_classes, channels, task, size=32):
    def build(data_cfg, split, transform):
        g = torch.Generator().manual_seed(0 if split == "train" else 1)
        n = 24
        x = torch.randn(n, channels, size, size, generator=g)
        if task == "multiclass":
            y = torch.eye(n_classes)[torch.randint(0, n_classes, (n,), generator=g)]
        else:
            y = (torch.rand(n, n_classes, generator=g) < 0.4).float()
        return _Labelled(x, y)
    return build


@pytest.fixture
def toy(request):
    names = []

    def _make(name, classes, task, channels=3, size=32):
        spec = DatasetSpec(name=name, description="synthetic", classes=classes, task=task,
                           in_channels=channels, required_keys=(),
                           build=_build(len(classes), channels, task, size),
                           transforms=lambda split: None)
        names.append(name)
        return register_dataset(spec)

    yield _make
    for name in names:
        registry._REGISTRY.pop(name, None)


def _cfg(tmp_path, spec, model="resnet18", monitor=None, epochs=2):
    cfg = {
        "data": {"name": spec.name, "num_workers": 0},
        "model": {"name": model, "pretrained": False},
        "training": {"epochs": epochs, "batch_size": 8, "lr": 0.001, "log_batch_every": 2,
                     "mixup_alpha": 0.2, "label_smoothing": 0.1},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
        "output": {"env": "test"},
    }
    if monitor is not None:
        cfg["monitor"] = monitor
    return cfg


def _train(tmp_path, monkeypatch, spec, monitor=None, model="resnet18", epochs=2, seed=7):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    random.seed(seed)                       # mixup decides with Python's random
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = _cfg(tmp_path, spec, model=model, monitor=monitor, epochs=epochs)
    trainer = (TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="t")
               .with_layers("confusion").build())
    dl_train = DataLoader(spec.dataset(cfg["data"], "train"), batch_size=8, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    dl_val = DataLoader(spec.dataset(cfg["data"], "val"), batch_size=8)
    trainer.fit(dl_train, dl_val, epochs=epochs)
    log_dir = tmp_path / "logs" / "test" / spec.name / "single" / model
    return trainer, log_dir


def _events(log_dir):
    return [json.loads(x) for x in (log_dir / "events_t.jsonl").read_text().splitlines()]


# ── The invariant ─────────────────────────────────────────────────────────────

def test_monitoring_does_not_change_the_training(tmp_path, monkeypatch, toy):
    spec = toy("toy_inv", ("a", "b", "c"), "multiclass")
    plain, _ = _train(tmp_path / "plain", monkeypatch, spec)
    rng_after_plain = torch.get_rng_state()
    random_after_plain = random.getstate()
    watched, _ = _train(tmp_path / "watched", monkeypatch, spec,
                        monitor={"enabled": True, "samples": 10, "gradcam": 3})
    rng_after_watched = torch.get_rng_state()

    for (n1, p1), (n2, p2) in zip(plain.model.named_parameters(),
                                  watched.model.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), f"{n1} differs with the monitor on"
    assert torch.equal(rng_after_plain, rng_after_watched)    # no global draws
    assert random.getstate() == random_after_plain


# ── Saved data follows the contract ───────────────────────────────────────────

def test_monitor_writes_the_inspection_contract(tmp_path, monkeypatch, toy):
    spec = toy("toy_contract", ("a", "b", "c", "d"), "multiclass")
    _, log_dir = _train(tmp_path, monkeypatch, spec,
                        monitor={"enabled": True, "samples": 12, "gradcam": 4,
                                 "thumbnail": 16})
    inspect = log_dir / "inspect_t"
    ms = np.load(inspect / "monitor_set.npz")
    assert ms["indices"].shape == (12,) and ms["targets"].shape == (12, 4)
    assert ms["thumbs"].shape == (12, 16, 16, 3) and ms["thumbs"].dtype == np.uint8
    assert list(ms["classes"]) == ["a", "b", "c", "d"] and str(ms["task"]) == "multiclass"
    assert len(ms["panel"]) == 4 and ms["targets"].sum(axis=1).tolist() == [1] * 12

    for epoch in (1, 2):
        ep = np.load(inspect / f"epoch_{epoch:03d}.npz")
        assert ep["probs"].shape == (12, 4) and ep["probs"].dtype == np.float16
        np.testing.assert_allclose(ep["probs"].astype(np.float32).sum(1), 1, atol=1e-2)
        assert ep["loss"].shape == (12,) and (ep["loss"] >= 0).all()
        assert ep["embeddings"].shape == (12, 512)              # resnet18 features
        assert ep["cam"].shape[0] == 4 and ep["cam"].dtype == np.uint8
        assert ep["cam_class"].shape == (4,)
        assert ep["val_probs"].shape == (24, 4) and ep["val_targets"].shape == (24, 4)
        assert float(ep["threshold"]) == pytest.approx(0.5)

    events = _events(log_dir)
    kinds = [e["type"] for e in events]
    start = next(e for e in events if e["type"] == "monitor_start")
    assert start["dir"] == "inspect_t" and start["samples"] == 12 and start["panel"] == 4
    assert start["embedding_dim"] == 512
    inspections = [e for e in events if e["type"] == "inspection"]
    assert [e["epoch"] for e in inspections] == [1, 2]
    assert inspections[0]["file"] == "inspect_t/epoch_001.npz"
    assert kinds.index("monitor_start") > kinds.index("run_start")
    layers = [e for e in events if e["type"] == "layers"]
    assert layers and {"name", "grad_norm", "weight_norm", "update_ratio"} <= set(
        layers[0]["groups"][0])
    hists = [e for e in events if e["type"] == "histograms"]
    assert [h["epoch"] for h in hists] == [1, 2]
    g = hists[0]["groups"][0]
    assert len(g["edges"]) == 33 and len(g["counts"]) == 32


def test_multilabel_vit_with_gradcam_and_threshold(tmp_path, monkeypatch, toy):
    spec = toy("toy_vit", ("x", "y", "z"), "multilabel", size=32)
    model = "vit_tiny_patch16_224"
    monkeypatch.setattr("src.models.vit.timm.create_model",
                        _small_vit(monkeypatch), raising=True)
    _, log_dir = _train(tmp_path, monkeypatch, spec, model=model, epochs=1,
                        monitor={"enabled": True, "samples": 8, "gradcam": 3,
                                 "layers": False, "embeddings": False})
    ep = np.load(log_dir / "inspect_t" / "epoch_001.npz")
    assert ep["cam"].shape == (3, 4, 4)                       # 32 px / patch 8 → 4×4
    assert ep["cam"].max() == 255                              # non-trivial maps
    assert "embeddings" not in ep.files
    assert 0 < float(ep["threshold"]) < 1
    assert not [e for e in _events(log_dir) if e["type"] in ("layers", "histograms")]


def _small_vit(monkeypatch):
    import timm
    real = timm.create_model

    def create(name, **kw):
        if name.startswith("vit_"):
            kw.update(img_size=32, patch_size=8, depth=2, num_heads=2, embed_dim=32)
        return real(name, **kw)
    return create


def test_monitor_off_by_default(tmp_path, monkeypatch, toy):
    spec = toy("toy_off", ("a", "b"), "multiclass")
    _, log_dir = _train(tmp_path, monkeypatch, spec)
    assert not (log_dir / "inspect_t").exists()
    assert not [e for e in _events(log_dir) if e["type"] == "monitor_start"]


def test_monitor_every_n_epochs_keeps_the_last(tmp_path, monkeypatch, toy):
    spec = toy("toy_every", ("a", "b"), "multiclass")
    _, log_dir = _train(tmp_path, monkeypatch, spec, epochs=3,
                        monitor={"enabled": True, "samples": 6, "every": 2, "gradcam": 0,
                                 "layers": False})
    files = sorted(p.name for p in (log_dir / "inspect_t").glob("epoch_*.npz"))
    assert files == ["epoch_002.npz", "epoch_003.npz"]


def test_a_broken_dataset_only_warns(tmp_path, monkeypatch, toy):
    spec = toy("toy_broken", ("a", "b"), "multiclass")
    monkeypatch.setattr("src.training.monitor.thumbnails",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("unreadable")))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, log_dir = _train(tmp_path, monkeypatch, spec,
                            monitor={"enabled": True, "samples": 6})
    assert any("unreadable" in str(w.message) for w in caught)
    assert _events(log_dir)[-1]["status"] == "completed"


# ── Helpers ───────────────────────────────────────────────────────────────────

def test_select_indices_is_seeded_and_covers_rare_classes():
    labels = torch.zeros(100, 3)
    labels[:95, 0] = 1
    labels[95:98, 1] = 1
    labels[98:, 2] = 1
    state = torch.get_rng_state()
    idx = select_indices(100, 12, seed=3, labels=labels)
    assert torch.equal(state, torch.get_rng_state())
    assert len(idx) == 12 and len(set(idx)) == 12
    assert set(range(95, 100)) <= set(idx.tolist())            # rare classes kept
    assert (idx == select_indices(100, 12, seed=3, labels=labels)).all()
    assert len(select_indices(5, 12, seed=0)) == 5


def test_panel_cycles_through_classes():
    targets = np.eye(3, dtype=np.uint8)[[0, 0, 0, 1, 1, 2]]
    assert sorted(targets[panel_positions(targets, 3)].argmax(1).tolist()) == [0, 1, 2]
    assert len(panel_positions(targets, 0)) == 0


def test_thumbnails_scale_each_image_and_replicate_gray():
    x = torch.linspace(-4, 9, 400).reshape(1, 1, 20, 20).repeat(2, 1, 1, 1)
    t = thumbnails(x, 8)
    assert t.shape == (2, 8, 8, 3) and t.dtype == np.uint8
    assert t.max() > 230 and t.min() < 25                     # stretched to its own range
    assert (t[..., 0] == t[..., 2]).all()


def test_layer_groups():
    assert layer_group("module.backbone.blocks.3.attn.qkv.weight") == "backbone.blocks.3"
    assert layer_group("base.head.1.bias") == "head.1"
    assert layer_group("backbone.cls_token") == "backbone.cls_token"
    assert layer_group("backbone.layer4.1.conv2.weight") == "backbone.layer4.1"


def test_gradcam_never_touches_parameter_gradients():
    from src.models.vit import TimmClassifier
    model = TimmClassifier("resnet18", num_classes=3, pretrained=False).eval()
    images = torch.randn(2, 3, 64, 64)
    maps = grad_cam(model, model, images, torch.tensor([0, 2]))
    assert maps.shape == (2, 2, 2) and maps.dtype == np.uint8
    assert maps.max() == 255
    assert all(p.grad is None for p in model.parameters())
