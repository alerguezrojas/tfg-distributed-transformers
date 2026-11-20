"""Política de checkpoints: carpeta por run, best/last, keep=all, escritura atómica y
compatibilidad con los ficheros antiguos. Todo en CPU."""
import os
import time

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.checkpoints import atomic_save, resolve_checkpoint
from src.training.decorators import TracingDecorator
from src.training.trainer import Trainer

N_CLASSES = 4


def _loader():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(16, 3, 4, 4, generator=g)
    y = (torch.rand(16, N_CLASSES, generator=g) < 0.4).float()
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _trainer(path, **kw):
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, N_CLASSES))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return Trainer(model, opt, None, torch.device("cpu"), checkpoint_dir=str(path), **kw)


def _fit(path, epochs=3, **kw):
    tr = TracingDecorator(_trainer(path, **kw))
    tr.fit(_loader(), _loader(), epochs)
    return tr


def test_best_last_keeps_two_files(tmp_path):
    _fit(tmp_path / "run", epochs=3)
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["best.pt", "last.pt"]
    last = torch.load(tmp_path / "run" / "last.pt", weights_only=True)
    assert last["epoch"] == 3          # last = final epoch, whether or not it was the best
    best = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    assert 1 <= best["epoch"] <= 3 and "model_state_dict" in best


def test_last_moves_every_epoch_and_best_stays(tmp_path):
    tr = _trainer(tmp_path)
    tr.save_checkpoint(2, {"f1": 0.5})
    tr.save_last_checkpoint(2, {"f1": 0.5})
    tr.save_last_checkpoint(3, {"f1": 0.4})       # worse epoch: last moves, best stays
    assert torch.load(tmp_path / "last.pt", weights_only=True)["epoch"] == 3
    assert torch.load(tmp_path / "best.pt", weights_only=True)["epoch"] == 2


def test_double_resume_keeps_the_global_epoch(tmp_path):
    """10 epochs, cut at 6, resume, cut at 2 of the second run: last.pt must say 8, not 2."""
    first = _trainer(tmp_path / "run1")
    first.save_last_checkpoint(6, {})
    second = _trainer(tmp_path / "run2")
    second.load_checkpoint(tmp_path / "run1")
    second.save_last_checkpoint(2, {})
    last = torch.load(tmp_path / "run2" / "last.pt", weights_only=True)
    assert last["epoch"] == 8
    third = _trainer(tmp_path / "run3")
    assert third.load_checkpoint(tmp_path / "run2")["epoch"] == 8


def test_resume_restores_best_metric_and_patience(tmp_path):
    tr = _trainer(tmp_path / "a")
    tr.save_last_checkpoint(3, {}, {"best_metric": 0.7, "epochs_no_improve": 2})
    tr2 = _trainer(tmp_path / "b")
    tr2.load_checkpoint(tmp_path / "a")
    assert tr2.resume_state == {"best_metric": 0.7, "epochs_no_improve": 2}
    seen = {}
    ctrl = TracingDecorator(tr2)
    orig = ctrl._should_stop
    ctrl._patience = 99
    ctrl._should_stop = lambda n: seen.setdefault("n", n) or orig(n)
    ctrl.fit(_loader(), _loader(), 1)
    assert seen["n"] >= 2        # the counter continued from 2, not from 0


def test_scaler_state_is_saved_and_restored(tmp_path):
    tr = _trainer(tmp_path / "a")
    tr._scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    tr.save_last_checkpoint(1, {})
    ckpt = torch.load(tmp_path / "a" / "last.pt", weights_only=True)
    assert ckpt["scaler_state_dict"]["scale"] == 1024.0
    tr2 = _trainer(tmp_path / "b")
    tr2._scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=2.0)
    tr2.load_checkpoint(tmp_path / "a")
    assert tr2._scaler.get_scale() == 1024.0


def test_failing_last_write_warns_and_training_goes_on(tmp_path):
    tr = _trainer(tmp_path)

    def full_disk(*a, **k):
        raise OSError("No space left on device")
    tr.save_last_checkpoint = full_disk
    with pytest.warns(RuntimeWarning, match="last.pt"):
        TracingDecorator(tr).fit(_loader(), _loader(), 2)
    assert (tmp_path / "best.pt").exists()


def test_failing_best_write_still_aborts(tmp_path):
    tr = _trainer(tmp_path)

    def full_disk(*a, **k):
        raise OSError("No space left on device")
    tr.save_checkpoint = full_disk
    with pytest.raises(OSError):
        TracingDecorator(tr).fit(_loader(), _loader(), 2)


def test_keep_all_keeps_the_old_scheme(tmp_path):
    tr = _trainer(tmp_path, checkpoint_keep="all")
    tr.save_checkpoint(1, {"f1": 0.5})
    tr.save_checkpoint(4, {"f1": 0.6})
    tr.save_last_checkpoint(5, {"f1": 0.1})
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "checkpoint_epoch_001.pt", "checkpoint_epoch_004.pt"]


def test_ddp_only_rank_zero_writes(tmp_path):
    from src.training.ddp_trainer import DDPTrainer
    tr = DDPTrainer.__new__(DDPTrainer)       # no process group: only the rank gate matters
    tr.rank = 1
    tr.save_checkpoint(1, {})
    tr.save_last_checkpoint(1, {})            # would fail on a missing model if it wrote
    assert list(tmp_path.iterdir()) == []


def test_atomic_save_leaves_no_partial_file_on_error(tmp_path, monkeypatch):
    target = tmp_path / "best.pt"
    atomic_save({"epoch": 1}, target)

    def boom(obj, path):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise OSError("No space left on device")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        atomic_save({"epoch": 2}, target)
    monkeypatch.undo()
    assert torch.load(target, weights_only=True)["epoch"] == 1   # intact
    assert not (tmp_path / "best.pt.tmp").exists()


@pytest.mark.parametrize("keep, expected", [
    (None, ["best.pt", "last.pt"]),
    ("all", None),
])
def test_builder_end_to_end_one_folder_per_run(tmp_path, monkeypatch, keep, expected):
    from src.data import registry
    from src.data.registry import register_dataset
    from src.data.spec import DatasetSpec
    from src.training.builder import TrainingSessionBuilder

    def build(data_cfg, split, transform):
        g = torch.Generator().manual_seed(0)
        return TensorDataset(torch.randn(16, 3, 32, 32, generator=g),
                             torch.eye(3)[torch.randint(0, 3, (16,), generator=g)])
    name = f"toy_ckpt_{keep or 'default'}"
    spec = register_dataset(DatasetSpec(
        name=name, description="synthetic", classes=("a", "b", "c"), task="multiclass",
        in_channels=3, required_keys=(), build=build, transforms=lambda split: None))
    monkeypatch.chdir(tmp_path)
    try:
        cfg = {"data": {"name": name, "num_workers": 0},
               "model": {"name": "resnet18", "pretrained": False},
               "training": {"epochs": 2, "batch_size": 8, "lr": 0.001},
               "checkpoint": {"dir": str(tmp_path / "ckpt")}, "output": {"env": "test"}}
        if keep:
            cfg["checkpoint"]["keep"] = keep
        trainer = TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="01012026_120000").build()
        trainer.fit(DataLoader(spec.dataset(cfg["data"], "train"), batch_size=8),
                    DataLoader(spec.dataset(cfg["data"], "val"), batch_size=8), epochs=2)
    finally:
        registry._REGISTRY.pop(name, None)
    run = tmp_path / "ckpt" / name / "single" / "resnet18" / "01012026_120000"
    names = sorted(p.name for p in run.iterdir())
    if expected:
        assert names == expected
    else:   # keep=all: old scheme, one file per improved epoch, no last.pt
        assert names and all(n.startswith("checkpoint_epoch_") for n in names)


# ── Compatibilidad y localización ─────────────────────────────────────────────

def test_old_checkpoint_file_still_loads_and_resolves(tmp_path):
    tr = _trainer(tmp_path, checkpoint_keep="all")
    tr.save_checkpoint(7, {"f1": 0.5})
    old = tmp_path / "checkpoint_epoch_007.pt"
    assert resolve_checkpoint(old) == old
    assert tr.load_checkpoint(old)["epoch"] == 7


def test_resolve_run_folder_prefers_best_or_last(tmp_path):
    tr = _trainer(tmp_path)
    tr.save_checkpoint(1, {})
    tr.save_last_checkpoint(2, {})
    assert resolve_checkpoint(tmp_path).name == "best.pt"
    assert resolve_checkpoint(tmp_path, prefer="last").name == "last.pt"
    assert tr.load_checkpoint(tmp_path)["epoch"] == 2   # resume from a folder = last.pt


def test_resolve_old_folder_takes_latest_epoch_file(tmp_path):
    tr = _trainer(tmp_path, checkpoint_keep="all")
    tr.save_checkpoint(1, {})
    time.sleep(0.02)
    tr.save_checkpoint(3, {})
    assert resolve_checkpoint(tmp_path).name == "checkpoint_epoch_003.pt"


def test_resolve_model_folder_picks_newest_run(tmp_path):
    for i, ts in enumerate(["09112026_100000", "19112026_100000"]):
        run = tmp_path / ts
        run.mkdir()
        atomic_save({"epoch": i}, run / "best.pt")
        os.utime(run / "best.pt", (1_000_000 + i * 100, 1_000_000 + i * 100))
    assert resolve_checkpoint(tmp_path).parent.name == "19112026_100000"


def test_resolve_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path)
