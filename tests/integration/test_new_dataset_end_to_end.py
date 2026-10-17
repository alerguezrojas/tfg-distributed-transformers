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


def _train(tmp_path, monkeypatch, spec, epochs=2):
    monkeypatch.chdir(tmp_path)              # the builder writes logs/ under cwd
    cfg = {
        "data": {"name": spec.name, "num_workers": 0},
        "model": {"name": "resnet18", "pretrained": False},
        "training": {"epochs": epochs, "batch_size": 8, "lr": 0.001,
                     "label_smoothing": 0.1, "mixup_alpha": 0.2},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
        "output": {"env": "test"},
    }
    trainer = (TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="t")
               .with_trace("simple").with_layers("confusion").build())
    train_dl = DataLoader(spec.dataset(cfg["data"], "train"), batch_size=8)
    val_dl = DataLoader(spec.dataset(cfg["data"], "val"), batch_size=8)
    trainer.fit(train_dl, val_dl, epochs=epochs)
    return trainer, tmp_path / "logs" / "test" / "single" / "resnet18"


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
