"""DDP must wrap the model with the index of ITS device, not with the global rank."""
from types import SimpleNamespace

import torch

import src.training.ddp_trainer as ddp_mod
from src.training.ddp_trainer import DDPTrainer, ddp_device_ids
from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer
from src.training.trainer import Trainer


def test_device_ids_use_the_device_index():
    assert ddp_device_ids(torch.device("cuda", 0)) == [0]
    assert ddp_device_ids(torch.device("cuda", 3)) == [3]


def test_device_ids_none_on_cpu():
    assert ddp_device_ids(torch.device("cpu")) is None


def _capture_ddp(monkeypatch, trainer_cls, rank, device, **kwargs):
    """Build the trainer with DDP and the base Trainer stubbed (no GPU, no process group)."""
    calls = []

    def fake_base_init(self, model, optimizer, scheduler, device, *a, **kw):
        self.model, self.device = model, device
        self.task = SimpleNamespace(sigmoid_outputs=True)
        self.criterion = None

    def fake_ddp(model, **kw):
        calls.append(kw)
        return model

    monkeypatch.setattr(Trainer, "__init__", fake_base_init)
    monkeypatch.setattr(ddp_mod, "DDP", fake_ddp)
    trainer_cls(torch.nn.Linear(2, 2), None, None, device, rank=rank, world_size=2, **kwargs)
    return calls


def test_rank_one_on_the_only_gpu_of_its_node_uses_cuda0(monkeypatch):
    # multinode, one GPU per node: global rank 1, local device cuda:0
    calls = _capture_ddp(monkeypatch, DDPTrainer, rank=1, device=torch.device("cuda", 0))
    assert calls == [{"device_ids": [0]}]


def test_heterogeneous_trainer_inherits_the_fix(monkeypatch):
    calls = _capture_ddp(monkeypatch, HeterogeneousDDPTrainer, rank=3,
                         device=torch.device("cuda", 1), local_batch_size=4)
    assert calls == [{"device_ids": [1]}]


def test_cpu_passes_no_device_ids(monkeypatch):
    calls = _capture_ddp(monkeypatch, DDPTrainer, rank=1, device=torch.device("cpu"))
    assert calls == [{}]
