"""El modelo efectivo (--model sobre la YAML) llega al manifiesto del run y al checkpoint.
Todo en CPU."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import registry
from src.data.registry import register_dataset
from src.data.spec import DatasetSpec
from src.run_manifest import manifest_path, read_manifest
from src.training.builder import TrainingSessionBuilder


@pytest.fixture
def toy_spec():
    def build(data_cfg, split, transform):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(16, 3, 32, 32, generator=g)
        y = (torch.rand(16, 3, generator=g) < 0.4).float()
        return TensorDataset(x, y)
    spec = register_dataset(DatasetSpec(
        name="toy_modelo", description="synthetic", classes=("a", "b", "c"), task="multilabel",
        in_channels=3, required_keys=(), build=build, transforms=lambda split: None))
    yield spec
    registry._REGISTRY.pop("toy_modelo", None)


def test_manifest_and_checkpoint_carry_the_overridden_model(tmp_path, monkeypatch, toy_spec):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "data": {"name": toy_spec.name, "num_workers": 0},
        "model": {"name": "resnet18", "pretrained": False},
        "training": {"epochs": 1, "batch_size": 8, "lr": 0.001},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
        "output": {"env": "test"},
    }
    trainer = (TrainingSessionBuilder(cfg, torch.device("cpu"), timestamp="t")
               .with_model("resnet10t").with_trace("simple").build())
    dl = DataLoader(toy_spec.dataset(cfg["data"], "train"), batch_size=8)
    trainer.fit(dl, dl, epochs=1)

    log_dir = tmp_path / "logs" / "test" / toy_spec.name / "single" / "resnet10t"
    manifest = read_manifest(manifest_path(log_dir, "t"))
    assert manifest["model"] == "resnet10t"
    assert manifest["config"]["model"]["name"] == "resnet10t"   # antes: el de la YAML
    assert cfg["model"]["name"] == "resnet18"                   # el cfg del llamador no se muta

    ckpts = list((tmp_path / "ckpt").rglob("best.pt"))
    assert len(ckpts) == 1
    assert torch.load(ckpts[0], weights_only=True)["model_name"] == "resnet10t"
