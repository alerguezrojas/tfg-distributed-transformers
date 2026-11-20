"""Rendimiento por batch: espera del DataLoader, tiempo de paso y muestras/s.

Los cuatro bucles de entrenamiento (Trainer —y con él DDP y paralelismo de modelo—,
heterogéneo y el bucle propio de DeepTracing) comparten el mismo StepTimer. Los valores
de reloj no son deterministas: se comprueba que existen y que son coherentes. Que no
alteran el entrenamiento lo fija test_behaviour_golden.py.
"""
import logging
import os
import time

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.training.decorators import DeepTracingDecorator
from src.training.ddp_trainer import DDPTrainer
from src.training.events import public_metrics
from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer
from src.training.model_parallel_trainer import ModelParallelTrainer
from src.training.trainer import Trainer

N, BS, C = 16, 4, 5
BATCH_KEYS = {"data_time_s", "step_time_s", "samples_per_s"}
EPOCH_KEYS = {"samples_per_s", "data_wait_frac"}


def _loader(n=N, bs=BS):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(n, 3, 8, 8, generator=g)
    y = (torch.rand(n, C, generator=g) < 0.3).float()
    return DataLoader(TensorDataset(x, y), batch_size=bs)


class _SlowDataset(Dataset):
    """Cada muestra tarda 15 ms en leerse: el entrenamiento espera por los datos."""

    def __len__(self):
        return N

    def __getitem__(self, i):
        time.sleep(0.015)
        return torch.randn(3, 8, 8), (torch.rand(C) < 0.3).float()


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, C))


def _base(cls, tmp_path, **kw):
    model = _model()
    return cls(model, torch.optim.AdamW(model.parameters(), lr=1e-2), None,
               torch.device("cpu"), checkpoint_dir=str(tmp_path), **kw)


@pytest.fixture
def gloo():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29498")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _single(tmp_path):
    return _base(Trainer, tmp_path)


def _model_parallel(tmp_path):
    return _base(ModelParallelTrainer, tmp_path)


def _ddp(tmp_path):
    return _base(DDPTrainer, tmp_path)


def _hetero(tmp_path):
    model = _model()
    return HeterogeneousDDPTrainer(
        model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-2),
        scheduler=None, device=torch.device("cpu"), checkpoint_dir=str(tmp_path),
        local_batch_size=BS)


def _deep(tmp_path):
    return DeepTracingDecorator(_base(Trainer, tmp_path), logger=logging.getLogger("tp"),
                                log_every=1000, features={"batch-table"})


BUILDERS = [_single, _model_parallel, _ddp, _hetero, _deep]


def _check_epoch(result):
    assert EPOCH_KEYS <= set(result)
    assert result["samples_per_s"] > 0
    assert 0.0 <= result["data_wait_frac"] <= 1.0


@pytest.mark.parametrize("make", BUILDERS, ids=lambda f: f.__name__.strip("_"))
def test_epoch_result_has_coherent_throughput(tmp_path, gloo, make):
    result = make(tmp_path).train_epoch(_loader())
    _check_epoch(result)
    # Sin hooks no se calculan métricas por batch, pero las del epoch sí.


@pytest.mark.parametrize("make", BUILDERS, ids=lambda f: f.__name__.strip("_"))
def test_batch_hooks_receive_timing(tmp_path, gloo, make):
    trainer = make(tmp_path)
    base = getattr(trainer, "_trainer", trainer)       # los hooks viven en el Trainer
    seen = []
    base.register_batch_hook(lambda e, b, n, mtr: seen.append(dict(mtr)))
    trainer.train_epoch(_loader())
    assert len(seen) == N // BS
    for mtr in seen:
        assert BATCH_KEYS <= set(mtr)
        assert mtr["data_time_s"] >= 0 and mtr["step_time_s"] > 0
        assert mtr["samples_per_s"] > 0
        # samples/s del batch = muestras / (espera + paso)
        assert mtr["samples_per_s"] == pytest.approx(
            BS / (mtr["data_time_s"] + mtr["step_time_s"]), rel=1e-6)


def test_epoch_throughput_is_consistent_with_the_batches(tmp_path):
    trainer = _single(tmp_path)
    seen = []
    trainer.register_batch_hook(lambda e, b, n, mtr: seen.append(dict(mtr)))
    result = trainer.train_epoch(_loader())
    data = sum(m["data_time_s"] for m in seen)
    step = sum(m["step_time_s"] for m in seen)
    assert result["data_wait_frac"] == pytest.approx(data / (data + step), rel=1e-6)
    assert result["samples_per_s"] == pytest.approx(N / (data + step), rel=1e-6)
    # El tiempo de pared de la época incluye todo lo anterior.
    assert data + step <= result["time"] + 1e-3


@pytest.mark.parametrize("make", [_single, _deep], ids=lambda f: f.__name__.strip("_"))
def test_a_slow_loader_shows_up_as_data_wait(tmp_path, make):
    slow = DataLoader(_SlowDataset(), batch_size=BS)    # num_workers=0: espera síncrona
    result = make(tmp_path).train_epoch(slow)
    _check_epoch(result)
    assert result["data_wait_frac"] > 0.5               # 60 ms de lectura por batch vs ~ms de paso


def test_a_fast_loader_barely_waits(tmp_path):
    trainer = _single(tmp_path)
    trainer.train_epoch(_loader())                      # calentamiento
    assert trainer.train_epoch(_loader())["data_wait_frac"] < 0.5


def test_hook_time_is_not_counted_as_data_wait(tmp_path):
    trainer = _single(tmp_path)
    trainer.register_batch_hook(lambda *a: time.sleep(0.05))   # hook lento (escribir a NFS…)
    trainer.train_epoch(_loader())                      # calentamiento
    result = trainer.train_epoch(_loader())
    assert result["data_wait_frac"] < 0.5               # 4 × 50 ms de hook no cuentan


def test_new_keys_reach_the_event_stream_but_not_the_epoch_csv(tmp_path):
    result = _single(tmp_path).train_epoch(_loader())
    public = public_metrics(result)
    assert EPOCH_KEYS <= set(public)                    # epoch_end.train las lleva solas
    assert all(isinstance(public[k], float) for k in EPOCH_KEYS)


def test_the_epoch_csv_keeps_its_columns(tmp_path):
    import pandas as pd
    from src.training.decorators import TracingDecorator
    csv = tmp_path / "epoch_metrics.csv"
    tr = TracingDecorator(_single(tmp_path), logger=logging.getLogger("tp-csv"),
                          epoch_csv=csv)
    tr.fit(_loader(), _loader(), epochs=1)
    cols = list(pd.read_csv(csv).columns)
    assert not {"samples_per_s", "data_wait_frac", "data_time_s", "step_time_s"} & set(cols)
