"""--resume en DDP: el muestreador debe seguir con la época siguiente, no repetir el barajado de la 1."""
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from src.training.ddp_trainer import DDPTrainer
from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer


@pytest.fixture
def gloo():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29487")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _trainer(cls, tmp_path, **kw):
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return cls(model=model, optimizer=opt, scheduler=None, device=torch.device("cpu"),
               checkpoint_dir=str(tmp_path / "ckpt"), **kw)


@pytest.mark.parametrize("cls,kw", [(DDPTrainer, {}), (HeterogeneousDDPTrainer,
                                                       {"local_batch_size": 4})])
def test_resumed_ddp_sampler_uses_the_next_epoch(tmp_path, gloo, cls, kw):
    first = _trainer(cls, tmp_path, **kw)
    first.save_last_checkpoint(3, {"val_f1": 0.5})
    resumed = _trainer(cls, tmp_path, **kw)
    resumed.load_checkpoint(str(tmp_path / "ckpt"))
    assert resumed._epoch == 3

    ds = TensorDataset(torch.randn(8, 3, 8, 8), torch.rand(8, 4).round())
    sampler = DistributedSampler(ds, num_replicas=1, rank=0)
    resumed.train_epoch(DataLoader(ds, batch_size=4, sampler=sampler))
    assert sampler.epoch == 4
