"""A dataset that is NOT BigEarthNet goes through the real builder and training
loop without touching any training code: register a spec, point the config at
it, train. Covers a multi-class (1-channel, like an X-ray) and a binary dataset.
"""
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import registry
from src.data.registry import register_dataset
from src.data.spec import DatasetSpec
from src.training.builder import TrainingSessionBuilder


def _synthetic(n_classes: int, channels: int, one_hot: bool):
    def build(data_cfg, split, transform):
        g = torch.Generator().manual_seed(0 if split == "train" else 1)
        n = 24
        x = torch.randn(n, channels, 32, 32, generator=g)
        if one_hot:
            y = torch.eye(n_classes)[torch.randint(0, n_classes, (n,), generator=g)]
        else:
            y = (torch.rand(n, n_classes, generator=g) < 0.4).float()
        return TensorDataset(x, y)
    return build


@pytest.fixture
def registered():
    names = []

    def _register(name, classes, task, channels):
        spec = DatasetSpec(
            name=name, description="synthetic", classes=classes, task=task,
            in_channels=channels, required_keys=(),
            build=_synthetic(len(classes), channels, one_hot=(task == "multiclass")),
            transforms=lambda split: None,
        )
        names.append(name)
        return register_dataset(spec)

    yield _register
    for name in names:
        registry._REGISTRY.pop(name, None)


def _train(tmp_path, monkeypatch, spec, epochs=2, inspect=()):
    monkeypatch.chdir(tmp_path)              # the builder writes logs/ under cwd
    cfg = {
        "data": {"name": spec.name, "num_workers": 0},
        "model": {"name": "resnet18", "pretrained": False},
        "training": {"epochs": epochs, "batch_size": 8, "lr": 0.001,
                     "label_smoothing": 0.1, "mixup_alpha": 0.2},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
        "output": {"env": "test"},
    }
    builder = (TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="t")
               .with_trace("simple").with_layers("confusion"))
    if inspect:
        builder = builder.with_inspect(*inspect)
    trainer = builder.build()
    train_dl = DataLoader(spec.dataset(cfg["data"], "train"), batch_size=8)
    val_dl = DataLoader(spec.dataset(cfg["data"], "val"), batch_size=8)
    trainer.fit(train_dl, val_dl, epochs=epochs)
    # Datasets other than BigEarthNet get their own level: logs/{env}/{dataset}/{mode}/{model}
    return trainer, tmp_path / "logs" / "test" / spec.name / "single" / "resnet18"


def test_model_summary_works_with_single_channel_inputs(tmp_path, monkeypatch, registered):
    spec = registered("toy_xray", ("normal", "abnormal"), "multiclass", channels=1)
    _, log_dir = _train(tmp_path, monkeypatch, spec, epochs=1, inspect=("model-summary",))
    log = (log_dir / "train_deep_t.log").read_text()
    assert "Total params" in log
    assert "resumen torchinfo omitido" not in log


def test_multiclass_single_channel_dataset_trains(tmp_path, monkeypatch, registered):
    spec = registered("toy_lesions", ("nevus", "melanoma", "keratosis", "other"),
                      "multiclass", channels=1)
    trainer, log_dir = _train(tmp_path, monkeypatch, spec)

    assert trainer.task.name == "multiclass"
    assert isinstance(trainer.criterion, torch.nn.CrossEntropyLoss)
    epochs = pd.read_csv(log_dir / "epoch_metrics_t.csv")
    assert len(epochs) == 2
    assert epochs["val_acc"].between(0, 1).all()
    conf = pd.read_csv(log_dir / "confusion_matrix_t.csv")
    assert set(conf["true_class"]) == set(spec.classes)
    # Rows of a multi-class confusion matrix are distributions (or empty).
    sums = conf[conf.epoch == 1].groupby("true_class")["value"].sum()
    assert all(s == pytest.approx(1.0, abs=1e-5) or s == 0 for s in sums)
    assert "threshold óptimo" not in (log_dir / "train_t.log").read_text()


def test_binary_dataset_trains_with_threshold_search(tmp_path, monkeypatch, registered):
    spec = registered("toy_tumour", ("tumour",), "binary", channels=3)
    trainer, log_dir = _train(tmp_path, monkeypatch, spec, epochs=1)

    assert trainer.task.name == "binary"
    assert isinstance(trainer.criterion, torch.nn.BCEWithLogitsLoss)
    per = pd.read_csv(log_dir / "perclass_metrics_t.csv")
    assert list(per["class_name"]) == ["tumour"]


# ── Every strategy must be buildable through the builder with a task ───────────

@pytest.fixture
def gloo():
    import os
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29495")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _innermost(trainer):
    while hasattr(trainer, "_trainer"):
        trainer = trainer._trainer
    return trainer


@pytest.mark.parametrize("hetero", [False, True], ids=["ddp", "heterogeneous"])
def test_distributed_strategies_build_with_the_dataset_task(tmp_path, monkeypatch,
                                                            registered, gloo, hetero):
    from src.training.ddp_trainer import DDPTrainer
    from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer
    spec = registered("toy_multilabel", ("a", "b", "c"), "multilabel", channels=3)
    monkeypatch.chdir(tmp_path)
    cfg = {
        "data": {"name": spec.name},
        "model": {"name": "resnet18", "pretrained": False},
        "training": {"epochs": 1, "batch_size": 8, "lr": 0.001},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
    }
    builder = TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="t",
                                     distributed=True).with_trace("off")
    if hetero:
        builder = builder.with_heterogeneous_ddp(8)
    trainer = builder.build()

    base = _innermost(trainer)
    assert type(base) is (HeterogeneousDDPTrainer if hetero else DDPTrainer)
    assert base.task.name == "multilabel"
    val = trainer.eval_epoch(DataLoader(spec.dataset(cfg["data"], "val"), batch_size=8))
    assert val["_preds"].shape[1] == 3


# ── Event stream (docs/eventos.md) ────────────────────────────────────────────

def _events(log_dir):
    import json
    return [json.loads(line) for line in (log_dir / "events_t.jsonl").read_text().splitlines()]


def test_training_writes_the_event_stream(tmp_path, monkeypatch, registered):
    spec = registered("toy_events", ("a", "b", "c"), "multiclass", channels=3)
    _, log_dir = _train(tmp_path, monkeypatch, spec, epochs=2)
    events = _events(log_dir)

    start = events[0]
    assert start["type"] == "run_start" and start["run_id"] == "t"
    assert (start["dataset"], start["task"], start["classes"]) == (
        "toy_events", "multiclass", ["a", "b", "c"])
    assert (start["model"], start["mode"], start["world_size"], start["epochs"]) == (
        "resnet18", "single", 1, 2)
    assert start["config"]["data"]["name"] == "toy_events"
    assert [e["epoch"] for e in events if e["type"] == "epoch_end"] == [1, 2]
    assert events[-1]["type"] == "run_end" and events[-1]["status"] == "completed"
    assert not [e for e in events if e["type"] == "batch"]     # no batch monitor → none


def test_batch_events_come_with_the_batch_monitor(tmp_path, monkeypatch, registered):
    spec = registered("toy_batches", ("tumour",), "binary", channels=3)
    monkeypatch.chdir(tmp_path)
    cfg = {
        "data": {"name": spec.name},
        "model": {"name": "resnet18", "pretrained": False},
        "training": {"epochs": 1, "batch_size": 8, "lr": 0.001, "log_batch_every": 2},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
        "output": {"env": "test"},
    }
    trainer = (TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="t")
               .with_layers("batch-monitor").build())
    dl = DataLoader(spec.dataset(cfg["data"], "train"), batch_size=8)   # 3 batches
    trainer.fit(dl, dl, epochs=1)
    log_dir = tmp_path / "logs" / "test" / spec.name / "single" / "resnet18"
    batches = [e for e in _events(log_dir) if e["type"] == "batch"]
    assert [b["batch"] for b in batches] == [2, 3]
    assert {"batch_loss", "running_loss", "lr"} <= set(batches[0]["metrics"])


@pytest.mark.parametrize("trace, rank", [("off", 0), ("simple", 1)])
def test_no_event_stream_without_trace_or_off_rank_zero(tmp_path, monkeypatch, registered,
                                                        trace, rank):
    spec = registered("toy_silent", ("a", "b"), "multiclass", channels=3)
    monkeypatch.chdir(tmp_path)
    cfg = {
        "data": {"name": spec.name},
        "model": {"name": "resnet18", "pretrained": False},
        "training": {"epochs": 1, "batch_size": 8, "lr": 0.001},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
        "output": {"env": "test"},
    }
    TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="t", rank=rank) \
        .with_trace(trace).build()
    assert not list(tmp_path.rglob("events_*.jsonl"))


# ── System sampler and per-batch throughput (docs/inspeccion.md) ──────────────

def _events_cfg(spec, tmp_path, **monitor):
    cfg = {
        "data": {"name": spec.name},
        "model": {"name": "resnet18", "pretrained": False},
        "training": {"epochs": 1, "batch_size": 8, "lr": 0.001, "log_batch_every": 1},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
        "output": {"env": "test"},
    }
    if monitor:
        cfg["monitor"] = monitor
    return cfg


def _sampler_listeners(trainer):
    from src.training.system_sampler import SystemSampler
    return [x for x in trainer._listeners if isinstance(x, SystemSampler)]


@pytest.mark.parametrize("monitor, expected", [
    ({}, 5),                                        # por defecto: cada 5 s, sin sección monitor
    ({"system_interval_s": 2}, 2),
    ({"system_interval_s": 0}, None),               # 0 lo desactiva
    ({"enabled": False}, 5),                        # sección monitor sin la clave
])
def test_builder_wires_the_system_sampler(tmp_path, monkeypatch, registered, monitor, expected):
    spec = registered("toy_system", ("a", "b"), "multiclass", channels=3)
    monkeypatch.chdir(tmp_path)
    cfg = _events_cfg(spec, tmp_path, **monitor)
    trainer = TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="t").with_trace(
        "simple").build()
    found = _sampler_listeners(trainer)
    assert [s.interval_s for s in found] == ([] if expected is None else [expected])
    if found:                                       # antes del sink: se para antes de run_end
        kinds = [type(x).__name__ for x in trainer._listeners]
        assert kinds.index("SystemSampler") < kinds.index("JsonlEventSink")


@pytest.mark.parametrize("trace, rank", [("off", 0), ("simple", 1)])
def test_no_system_sampler_without_event_stream(tmp_path, monkeypatch, registered, trace, rank):
    spec = registered("toy_system_off", ("a", "b"), "multiclass", channels=3)
    monkeypatch.chdir(tmp_path)
    trainer = TrainingSessionBuilder(_events_cfg(spec, tmp_path), torch.device("cpu"),
                                     timestamp="t", rank=rank).with_trace(trace).build()
    assert _sampler_listeners(trainer) == []


def test_run_writes_system_events_and_throughput_fields(tmp_path, monkeypatch, registered):
    spec = registered("toy_throughput", ("a", "b", "c"), "multiclass", channels=3)
    monkeypatch.chdir(tmp_path)
    cfg = _events_cfg(spec, tmp_path, system_interval_s=0.01)
    trainer = (TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="t")
               .with_layers("batch-monitor").build())
    dl = DataLoader(spec.dataset(cfg["data"], "train"), batch_size=8)
    trainer.fit(dl, dl, epochs=2)
    log_dir = tmp_path / "logs" / "test" / spec.name / "single" / "resnet18"
    events = _events(log_dir)

    batches = [e for e in events if e["type"] == "batch"]
    assert batches
    for b in batches:
        assert {"data_time_s", "step_time_s", "samples_per_s"} <= set(b["metrics"])
    for e in (e for e in events if e["type"] == "epoch_end"):
        assert e["train"]["samples_per_s"] > 0
        assert 0.0 <= e["train"]["data_wait_frac"] <= 1.0
    assert events[-1]["type"] == "run_end"          # el sampler para antes del run_end
    for s in (e for e in events if e["type"] == "system"):
        assert {"gpus", "cpu_pct", "ram_used_pct", "proc_cpu_pct", "proc_rss_mb"} <= set(s)
