"""Acumulación de gradientes con DDP (gloo, world_size=2): solo el último micro-lote del
grupo sincroniza (``no_sync`` en los demás) y el resultado es el del lote global."""
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.ddp_trainer import DDPTrainer
from src.training.trainer import Trainer

N_CLASSES = 5
ACCUM = 2


def _data():
    g = torch.Generator().manual_seed(0)
    return torch.randn(8, 3, 4, 4, generator=g), (torch.rand(8, N_CLASSES, generator=g) > 0.5).float()


def _model():
    torch.manual_seed(1)
    return nn.Sequential(nn.Flatten(), nn.Linear(48, N_CLASSES))


def _worker(rank, port, tmp):
    import os
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        x, y = _data()
        xs, ys = x[rank * 4:(rank + 1) * 4], y[rank * 4:(rank + 1) * 4]   # 2 micro-lotes de 2
        model = _model()
        tr = DDPTrainer(model, torch.optim.SGD(model.parameters(), lr=0.1), None,
                        torch.device("cpu"), checkpoint_dir=f"{tmp}/ck{rank}", rank=rank,
                        world_size=2, grad_accum_steps=ACCUM)
        calls = {"no_sync": 0}
        original = tr.model.no_sync

        def counting():
            calls["no_sync"] += 1
            return original()
        tr.model.no_sync = counting
        tr.train_epoch(DataLoader(TensorDataset(xs, ys), batch_size=2))
        weights = torch.cat([p.detach().flatten() for p in tr.model.parameters()])
        torch.save({"w": weights, "no_sync": calls["no_sync"]}, f"{tmp}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_ddp_accumulation_syncs_once_per_group_and_matches_global_batch(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_worker, args=(port, str(tmp_path)), nprocs=2, join=True)
    r0 = torch.load(tmp_path / "r0.pt", weights_only=True)
    r1 = torch.load(tmp_path / "r1.pt", weights_only=True)
    # 2 micro-lotes por rank, k=2: el primero va con no_sync, el segundo sincroniza
    assert r0["no_sync"] == r1["no_sync"] == 1
    assert torch.allclose(r0["w"], r1["w"], atol=1e-6)
    # Referencia: un solo proceso, un paso con las 8 muestras
    x, y = _data()
    model = _model()
    ref = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), None,
                  torch.device("cpu"), checkpoint_dir=str(tmp_path / "ref"))
    ref.train_epoch(DataLoader(TensorDataset(x, y), batch_size=8))
    expected = torch.cat([p.detach().flatten() for p in ref.model.parameters()])
    assert torch.allclose(r0["w"], expected, atol=1e-6)
