"""DeepTracingDecorator con batch-table posee su bucle: debe seguir alimentando los
batch hooks del Trainer, o BatchMonitorDecorator escribiría un CSV vacío."""

import logging

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.decorators import DeepTracingDecorator
from src.training.decorators.batch_monitor import BatchMonitorDecorator
from src.training.trainer import Trainer


def _loader(n: int = 16, bs: int = 4) -> DataLoader:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(n, 3, 8, 8, generator=g)
    y = (torch.rand(n, 19, generator=g) < 0.3).float()
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def _stack(tmp_path, features):
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 19))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    base = Trainer(model, opt, None, torch.device("cpu"), checkpoint_dir=str(tmp_path / "ckpt"))
    monitor = BatchMonitorDecorator(base, output_dir=str(tmp_path), timestamp="t")
    deep = DeepTracingDecorator(monitor, logger=logging.getLogger("deep-bm"),
                                log_every=1000, features=features)
    return deep, monitor


def test_batch_table_alimenta_batch_monitor(tmp_path):
    deep, monitor = _stack(tmp_path, {"batch-table"})
    deep.train_epoch(_loader())
    df = pd.read_csv(monitor.batch_csv_path)
    assert len(df) == 4                       # 16 muestras / bs=4
    assert list(df["batch"]) == [1, 2, 3, 4]
    assert df["batch_loss"].notna().all() and df["batch_f1"].notna().all()


def test_batch_monitor_con_deep_sin_batch_table(tmp_path):
    # Regresión: sin batch-table se delega en la cadena normal y el CSV ya se llenaba.
    deep, monitor = _stack(tmp_path, {"grad-monitor"})
    deep.train_epoch(_loader())
    assert len(pd.read_csv(monitor.batch_csv_path)) == 4


def test_batch_table_numera_las_epocas_al_pasar_por_fit(tmp_path):
    deep, monitor = _stack(tmp_path, {"batch-table"})
    deep.fit(_loader(), _loader(), epochs=2)
    df = pd.read_csv(monitor.batch_csv_path)
    assert list(df["epoch"]) == [1] * 4 + [2] * 4
