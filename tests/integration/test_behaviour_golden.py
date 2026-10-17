"""Characterization ("golden") tests: pin the exact numbers the training core
produced BEFORE the dataset/task abstraction, so the refactor provably changes
nothing for BigEarthNet (multi-label).

The expected values below were recorded from the pre-refactor code
(develop @ 4d74549) on CPU with fixed seeds. If one of these fails, behaviour
changed — investigate before touching the numbers.
"""
from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

N_CLASSES = 19
REL = 1e-6


# ── Deterministic fixtures ────────────────────────────────────────────────────

def _seed(s: int = 0) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _loader(n: int = 32, bs: int = 8) -> DataLoader:
    g = torch.Generator().manual_seed(123)
    x = torch.randn(n, 3, 8, 8, generator=g)
    y = (torch.rand(n, N_CLASSES, generator=g) < 0.3).float()
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=False)


def _model() -> nn.Module:
    torch.manual_seed(7)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, N_CLASSES))


def _round(d: dict) -> dict:
    return {k: round(float(v), 6) for k, v in d.items()
            if not k.startswith("_") and k != "time"}


# ── Trainer (single) ──────────────────────────────────────────────────────────

def _run_trainer(tmp_path, **kw):
    from src.training.trainer import Trainer
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tr = Trainer(model, opt, None, torch.device("cpu"),
                 checkpoint_dir=str(tmp_path), **kw)
    _seed(0)
    train = tr.train_epoch(_loader())
    val = tr.eval_epoch(_loader())
    return tr, train, val


def test_trainer_plain_bce_matches_golden(tmp_path):
    _, train, val = _run_trainer(tmp_path)
    assert _round(train) == pytest.approx(GOLDEN_PLAIN_TRAIN, rel=REL)
    assert _round(val) == pytest.approx(GOLDEN_PLAIN_VAL, rel=REL)
    assert val["_optimal_threshold"] == pytest.approx(GOLDEN_PLAIN_THR)
    assert val["_f1_at_optimal_threshold"] == pytest.approx(GOLDEN_PLAIN_F1_THR, rel=REL)
    assert int(val["_preds"].sum()) == GOLDEN_PLAIN_NPRED


def test_trainer_smoothing_mixup_matches_golden(tmp_path):
    _, train, val = _run_trainer(tmp_path, label_smoothing=0.1, mixup_alpha=0.2)
    assert _round(train) == pytest.approx(GOLDEN_REG_TRAIN, rel=REL)
    assert _round(val) == pytest.approx(GOLDEN_REG_VAL, rel=REL)
    assert val["_optimal_threshold"] == pytest.approx(GOLDEN_REG_THR)


def test_batch_hook_metrics_match_golden(tmp_path):
    from src.training.trainer import Trainer
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tr = Trainer(model, opt, None, torch.device("cpu"), checkpoint_dir=str(tmp_path))
    seen = []
    tr.register_batch_hook(lambda e, b, n, mtr: seen.append(_round(mtr)))
    _seed(0)
    tr.train_epoch(_loader())
    assert seen[-1] == pytest.approx(GOLDEN_LAST_BATCH, rel=REL)


# ── Deep tracing loop ─────────────────────────────────────────────────────────

def test_deep_tracing_train_loop_matches_golden(tmp_path):
    import logging
    from src.training.trainer import Trainer
    from src.training.decorators import DeepTracingDecorator
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    base = Trainer(model, opt, None, torch.device("cpu"), checkpoint_dir=str(tmp_path),
                   label_smoothing=0.1, mixup_alpha=0.2)
    deep = DeepTracingDecorator(base, logger=logging.getLogger("golden"),
                                log_every=1000, features={"batch-table"})
    _seed(0)
    train = deep.train_epoch(_loader())
    assert _round(train) == pytest.approx(GOLDEN_DEEP_TRAIN, rel=REL)


# ── Heterogeneous DDP (world_size=1, gloo) ────────────────────────────────────

@pytest.fixture
def gloo():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29497")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def test_heterogeneous_trainer_matches_golden(tmp_path, gloo):
    from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    tr = HeterogeneousDDPTrainer(model=model, optimizer=opt, scheduler=None,
                                 device=torch.device("cpu"),
                                 checkpoint_dir=str(tmp_path),
                                 label_smoothing=0.1, mixup_alpha=0.2,
                                 local_batch_size=8)
    _seed(0)
    train = tr.train_epoch(_loader())
    val = tr.eval_epoch(_loader())
    assert _round(train) == pytest.approx(GOLDEN_HETERO_TRAIN, rel=REL)
    assert _round(val) == pytest.approx(GOLDEN_HETERO_VAL, rel=REL)


# ── Confusion-matrix decorator CSVs ───────────────────────────────────────────

def test_confusion_decorator_csvs_match_golden(tmp_path):
    from src.training.trainer import Trainer
    from src.training.decorators import ConfusionMatrixDecorator
    from src.data.bigearthnet import BIGEARTHNET
    model = _model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    base = Trainer(model, opt, None, torch.device("cpu"), checkpoint_dir=str(tmp_path))
    dec = ConfusionMatrixDecorator(base, output_dir=str(tmp_path), timestamp="g",
                                   csv_dir=str(tmp_path), class_names=BIGEARTHNET.classes)
    _seed(0)
    dec.eval_epoch(_loader())
    per = pd.read_csv(tmp_path / "perclass_metrics_g.csv")
    conf = pd.read_csv(tmp_path / "confusion_matrix_g.csv")
    assert list(per["class_name"]) == GOLDEN_CLASS_ORDER
    assert round(float(per["f1"].sum()), 6) == pytest.approx(GOLDEN_PERCLASS_F1_SUM, rel=REL)
    assert len(conf) == N_CLASSES * N_CLASSES
    assert round(float(conf["value"].sum()), 6) == pytest.approx(GOLDEN_CONF_SUM, rel=REL)


# ── BigEarthNet reader + pos_weight from metadata ─────────────────────────────

def _fake_bigearthnet(tmp_path):
    """Two synthetic patches on disk with the real BigEarthNet layout."""
    import rasterio
    from rasterio.transform import from_origin
    rng = np.random.default_rng(5)
    rows = []
    for i, labels in enumerate([["Urban fabric", "Marine waters"], ["Arable land"]]):
        pid = f"S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_{i:02d}_{i:02d}"
        scene = "_".join(pid.rsplit("_", 2)[:-2])
        d = tmp_path / "root" / scene / pid
        d.mkdir(parents=True)
        for band in ("B04", "B03", "B02"):
            arr = rng.integers(0, 12000, size=(120, 120)).astype("uint16")
            with rasterio.open(d / f"{pid}_{band}.tif", "w", driver="GTiff", height=120,
                               width=120, count=1, dtype="uint16",
                               transform=from_origin(0, 0, 10, 10)) as dst:
                dst.write(arr, 1)
        rows.append({"patch_id": pid, "labels": labels, "split": "train"})
    rows.append({"patch_id": "unused", "labels": ["Pastures"], "split": "validation"})
    meta = tmp_path / "metadata.parquet"
    pd.DataFrame(rows).to_parquet(meta)
    return tmp_path / "root", meta


def test_bigearthnet_reader_matches_golden(tmp_path):
    from src.data.bigearthnet import BigEarthNetDataset, get_transforms
    root, meta = _fake_bigearthnet(tmp_path)
    raw = BigEarthNetDataset(str(root), str(meta), split="train")
    img, lab = raw[0]
    assert tuple(img.shape) == (3, 120, 120)
    assert round(float(img.sum()), 3) == pytest.approx(GOLDEN_RAW_IMG_SUM, rel=REL)
    assert torch.nonzero(lab).flatten().tolist() == [0, 18]
    tfm = BigEarthNetDataset(str(root), str(meta), split="val",
                             transform=get_transforms("val"))
    assert len(tfm) == 1
    _seed(0)
    tr = BigEarthNetDataset(str(root), str(meta), split="train",
                            transform=get_transforms("val"))
    x, _ = tr[1]
    assert tuple(x.shape) == (3, 224, 224)
    assert round(float(x.sum()), 2) == pytest.approx(GOLDEN_VAL_TFM_SUM, rel=1e-5)


def test_pos_weight_auto_matches_golden(tmp_path):
    from src.data.bigearthnet import BIGEARTHNET
    from src.training.losses import pos_weight_from_counts
    _, meta = _fake_bigearthnet(tmp_path)
    pw = pos_weight_from_counts(*BIGEARTHNET.class_counts({"metadata": str(meta)}, "train"))
    assert pw.tolist() == pytest.approx(GOLDEN_POS_WEIGHT)


# ── Threshold search actually moving off 0.5 (single + DDP global search) ─────

def _low_bias_model() -> nn.Module:
    model = _model()
    with torch.no_grad():
        model[1].bias.fill_(-1.2)          # low probabilities → optimum below 0.5
    return model


def test_threshold_search_single_matches_golden(tmp_path):
    from src.training.trainer import Trainer
    model = _low_bias_model()
    tr = Trainer(model, torch.optim.AdamW(model.parameters()), None,
                 torch.device("cpu"), checkpoint_dir=str(tmp_path))
    val = tr.eval_epoch(_loader())
    assert _round(val) == pytest.approx(GOLDEN_LOW_VAL, rel=REL)
    assert val["_optimal_threshold"] == pytest.approx(GOLDEN_LOW_THR)
    assert val["_f1_at_optimal_threshold"] == pytest.approx(GOLDEN_LOW_F1_THR, rel=REL)


def test_threshold_search_ddp_matches_golden(tmp_path, gloo):
    from src.training.ddp_trainer import DDPTrainer
    model = _low_bias_model()
    tr = DDPTrainer(model, torch.optim.AdamW(model.parameters()), None,
                    torch.device("cpu"), checkpoint_dir=str(tmp_path))
    val = tr.eval_epoch(_loader())
    assert _round(val) == pytest.approx(GOLDEN_LOW_VAL, rel=REL)
    assert val["_optimal_threshold"] == pytest.approx(GOLDEN_LOW_THR)


# ── Recorded golden values (pre-refactor, develop @ 4d74549) ─────────────────

GOLDEN_PLAIN_TRAIN = {'loss': 0.752266, 'f1': 0.365506, 'accuracy': 0.491776}
GOLDEN_PLAIN_VAL = {
    'loss': 0.496806,
    'f1': 0.722537,
    'accuracy': 0.825658,
    'precision': 0.680566,
    'recall': 0.794964,
}
GOLDEN_PLAIN_THR = 0.5
GOLDEN_PLAIN_F1_THR = 0.722537
GOLDEN_PLAIN_NPRED = 217
GOLDEN_REG_TRAIN = {'loss': 0.750546, 'f1': 0.367253, 'accuracy': 0.480263}
GOLDEN_REG_VAL = {
    'loss': 0.497263,
    'f1': 0.722185,
    'accuracy': 0.827303,
    'precision': 0.681742,
    'recall': 0.790977,
}
GOLDEN_REG_THR = 0.5
GOLDEN_LAST_BATCH = {
    'running_loss': 0.752266,
    'batch_loss': 0.748924,
    'lr': 0.01,
    'batch_f1': 0.36249,
    'batch_acc': 0.493421,
    'batch_prec': 0.323684,
}
GOLDEN_DEEP_TRAIN = {'loss': 0.750546, 'f1': 0.367253, 'accuracy': 0.480263}
GOLDEN_HETERO_TRAIN = {'loss': 0.749565, 'f1': 0.340758, 'accuracy': 0.46875}
GOLDEN_HETERO_VAL = {
    'loss': 0.503259,
    'f1': 0.713163,
    'accuracy': 0.817434,
    'precision': 0.659051,
    'recall': 0.794441,
}
GOLDEN_CLASS_ORDER = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland, shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters',
]
GOLDEN_PERCLASS_F1_SUM = 7.328138
GOLDEN_CONF_SUM = 191.265448
GOLDEN_RAW_IMG_SUM = 25213.754
GOLDEN_VAL_TFM_SUM = 88235.38
GOLDEN_POS_WEIGHT = [
    1.0,
    100.0,
    1.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    1.0,
]
GOLDEN_LOW_VAL = {
    'loss': 0.675723,
    'f1': 0.048423,
    'accuracy': 0.679276,
    'precision': 0.236842,
    'recall': 0.027388,
}
GOLDEN_LOW_THR = 0.1
GOLDEN_LOW_F1_THR = 0.467966
