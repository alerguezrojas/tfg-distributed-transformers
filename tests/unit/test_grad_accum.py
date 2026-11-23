"""Acumulación de gradientes (training.grad_accum_steps) y activation checkpointing
(training.grad_checkpointing): equivalencia numérica, validación y errores claros.

La equivalencia vale para un modelo sin dropout y sin BatchNorm en modo train: BatchNorm
normaliza con las estadísticas de cada micro-lote, así que k micro-lotes de b no dan el
gradiente de un lote de k·b (la acumulación sigue siendo válida, pero no es equivalente).
"""
import logging
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.vit import TimmClassifier
from src.training.config_validator import ConfigError, validate_config
from src.training.losses import memory_log_fields
from src.training.trainer import Trainer

N_CLASSES = 5


def _data(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 4, 4, generator=g), (torch.rand(n, N_CLASSES, generator=g) > 0.5).float()


def _loader(n, bs, seed=0):
    x, y = _data(n, seed)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=False)


def _trainer(tmp_path, accum=1, **kw):
    torch.manual_seed(1)
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, N_CLASSES))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return Trainer(model, opt, None, torch.device("cpu"), checkpoint_dir=str(tmp_path),
                   grad_accum_steps=accum, **kw)


def _weights(tr):
    return torch.cat([p.detach().flatten() for p in tr.model.parameters()])


class _CountingScaler:
    """GradScaler simulado (en CPU no hay AMP): cuenta las llamadas del paso."""

    def __init__(self):
        self.calls = []

    def scale(self, loss):
        return loss

    def unscale_(self, _opt):
        self.calls.append("unscale")

    def step(self, opt):
        self.calls.append("step")
        opt.step()

    def update(self):
        self.calls.append("update")

    def is_enabled(self):
        return False


def test_accumulating_k_microbatches_equals_one_big_batch(tmp_path):
    big = _trainer(tmp_path / "a", accum=1)
    big.train_epoch(_loader(8, 8))
    acc = _trainer(tmp_path / "b", accum=4)
    acc.train_epoch(_loader(8, 2))          # 4 micro-lotes de 2 -> 1 paso
    assert torch.allclose(_weights(big), _weights(acc), atol=1e-6)


def test_without_accumulation_the_same_data_gives_other_weights(tmp_path):
    """Sin acumular son 4 pasos: pesos distintos (el test de arriba no es trivial)."""
    big = _trainer(tmp_path / "a", accum=1)
    big.train_epoch(_loader(8, 8))
    four_steps = _trainer(tmp_path / "b", accum=1)
    four_steps.train_epoch(_loader(8, 2))
    assert not torch.allclose(_weights(big), _weights(four_steps), atol=1e-4)


def test_grad_clip_applies_to_the_accumulated_gradient(tmp_path):
    big = _trainer(tmp_path / "a", accum=1, grad_clip=0.05)
    big.train_epoch(_loader(8, 8))
    acc = _trainer(tmp_path / "b", accum=4, grad_clip=0.05)
    acc.train_epoch(_loader(8, 2))
    assert torch.allclose(_weights(big), _weights(acc), atol=1e-6)


def test_scaler_unscale_step_update_only_at_the_step(tmp_path):
    tr = _trainer(tmp_path, accum=4, grad_clip=1.0)
    tr._scaler = _CountingScaler()
    tr.train_epoch(_loader(16, 2))          # 8 micro-lotes -> 2 pasos
    assert tr._scaler.calls == ["unscale", "step", "update"] * 2


def test_incomplete_last_group_also_steps_and_averages_its_own_size(tmp_path):
    # 6 micro-lotes con k=4: grupos de 4 y de 2. El de 2 se promedia entre 2, no entre 4.
    tr = _trainer(tmp_path / "a", accum=4)
    tr._scaler = _CountingScaler()
    tr.train_epoch(_loader(12, 2))
    assert tr._scaler.calls.count("step") == 2
    ref = _trainer(tmp_path / "b", accum=1)
    x, y = _data(12)
    for part in (slice(0, 8), slice(8, 12)):          # grupo de 4 micro-lotes y grupo de 2
        ref.optimizer.zero_grad()
        ref.criterion(ref.model(x[part]), y[part]).backward()
        ref.optimizer.step()
    assert torch.allclose(_weights(tr), _weights(ref), atol=1e-6)


def test_hooks_still_see_every_microbatch(tmp_path):
    tr = _trainer(tmp_path, accum=3)
    seen = []
    tr.register_batch_hook(lambda ep, i, n, m: seen.append((i, n)))
    tr.train_epoch(_loader(10, 2))                    # 5 micro-lotes
    assert seen == [(i, 5) for i in range(1, 6)]


def test_epoch_loss_is_not_scaled_by_accumulation(tmp_path):
    # Con lr=0 los pesos no cambian: la pérdida de época es idéntica con y sin acumulación.
    a = _trainer(tmp_path / "a", accum=1)
    b = _trainer(tmp_path / "b", accum=4)
    for tr in (a, b):
        tr.optimizer.param_groups[0]["lr"] = 0.0
    assert b.train_epoch(_loader(8, 2))["loss"] == pytest.approx(
        a.train_epoch(_loader(8, 2))["loss"], rel=1e-6)


def test_default_leaves_the_step_per_batch(tmp_path):
    tr = _trainer(tmp_path)
    tr._scaler = _CountingScaler()
    tr.train_epoch(_loader(8, 2))
    assert tr._scaler.calls == ["step", "update"] * 4


def test_group_helper():
    tr = Trainer.__new__(Trainer)
    tr.grad_accum_steps = 4
    assert [tr._accum_group(i, 6) for i in range(1, 7)] == [
        (4, False), (4, False), (4, False), (4, True), (2, False), (2, True)]


# ── Validación y línea de configuración ──────────────────────────────────────

def _cfg(**train):
    return {"data": {"root": "/r", "metadata": "/m.parquet"},
            "model": {"name": "vit_tiny_patch16_224"},
            "training": {"epochs": 1, "batch_size": 4, "lr": 0.001, **train}}


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True])
def test_validator_rejects_bad_grad_accum_steps(bad):
    with pytest.raises(ConfigError, match="grad_accum_steps"):
        validate_config(_cfg(grad_accum_steps=bad))


def test_validator_accepts_good_values():
    validate_config(_cfg(grad_accum_steps=4, grad_checkpointing=True))


@pytest.mark.parametrize("bad", ["yes", 1, None])
def test_validator_rejects_non_bool_grad_checkpointing(bad):
    with pytest.raises(ConfigError, match="grad_checkpointing"):
        validate_config(_cfg(grad_checkpointing=bad))


def test_validator_rejects_checkpointing_with_model_parallel():
    with pytest.raises(ConfigError, match="model parallelism"):
        validate_config(_cfg(grad_checkpointing=True), model_parallel=True)


def test_config_line_gains_keys_only_when_active():
    assert memory_log_fields({}) == ""
    assert memory_log_fields({"grad_accum_steps": 1, "grad_checkpointing": False}) == ""
    assert memory_log_fields({"grad_accum_steps": 4}) == " | accum=4"
    assert memory_log_fields({"grad_accum_steps": 2, "grad_checkpointing": True}) == \
        " | accum=2 | ckpt=true"


def test_config_line_with_new_keys_still_parses_in_the_manifest():
    from src.run_manifest import parse_config_line
    line = ("Configuración: modelo=m | batch=8/GPU (global=8) | epochs=2 | lr=0.001 | "
            "precision=fp32 | loss=bce | train=10 | val=4 | dataset=panda | seed=1 | "
            "accum=4 | ckpt=true")
    fields = parse_config_line(line)
    assert fields["epochs"] == "2" and fields["accum"] == "4" and fields["ckpt"] == "true"


# ── Estrategias que no lo soportan ───────────────────────────────────────────

def test_heterogeneous_rejects_accumulation(tmp_path):
    from src.training.heterogeneous_ddp_trainer import HeterogeneousDDPTrainer
    model = nn.Sequential(nn.Flatten(), nn.Linear(48, N_CLASSES))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29491")
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        with pytest.raises(NotImplementedError, match="grad_accum_steps"):
            HeterogeneousDDPTrainer(model, opt, None, torch.device("cpu"),
                                    checkpoint_dir=str(tmp_path), rank=0, world_size=1,
                                    local_batch_size=2, grad_accum_steps=2)
    finally:
        dist.destroy_process_group()


def test_deep_batch_table_rejects_accumulation(tmp_path):
    from src.training.decorators.deep_tracing import DeepTracingDecorator
    tr = _trainer(tmp_path, accum=2)
    with pytest.raises(ValueError, match="grad_accum_steps"):
        DeepTracingDecorator(tr, logging.getLogger("t"), features={"batch-table"})


# ── Activation checkpointing ─────────────────────────────────────────────────

def _grads(model, x, y):
    model.zero_grad()
    nn.BCEWithLogitsLoss()(model(x), y).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


@pytest.mark.parametrize("name,kw", [("vit_tiny_patch16_224", {"img_size": 32}),
                                      ("resnet18", {})])
def test_grad_checkpointing_gives_the_same_gradient(name, kw):
    torch.manual_seed(0)
    plain = TimmClassifier(name, num_classes=N_CLASSES, pretrained=False, dropout=0.0, **kw)
    ckpt = TimmClassifier(name, num_classes=N_CLASSES, pretrained=False, dropout=0.0, **kw)
    ckpt.load_state_dict(plain.state_dict())
    ckpt.set_grad_checkpointing(True)
    plain.train()
    ckpt.train()
    side = kw.get("img_size", 64)
    x = torch.randn(4, 3, side, side)
    y = (torch.rand(4, N_CLASSES) > 0.5).float()
    assert torch.allclose(_grads(plain, x, y), _grads(ckpt, x, y), atol=1e-5, rtol=1e-4)


def test_grad_checkpointing_off_by_default_and_toggles():
    m = TimmClassifier("vit_tiny_patch16_224", num_classes=N_CLASSES, pretrained=False, img_size=32)
    assert not m.backbone.grad_checkpointing
    m.set_grad_checkpointing(True)
    assert m.backbone.grad_checkpointing
    m.set_grad_checkpointing(False)
    assert not m.backbone.grad_checkpointing


def test_backbone_without_checkpointing_gives_clear_error():
    m = TimmClassifier("resnet18", num_classes=N_CLASSES, pretrained=False)
    m.backbone = nn.Identity()
    with pytest.raises(ValueError, match="activation checkpointing"):
        m.set_grad_checkpointing(True)


def test_hf_encoder_without_support_gives_clear_error():
    from src.models.backbones import HFEncoder
    enc = HFEncoder.__new__(HFEncoder)
    nn.Module.__init__(enc)
    enc.model = nn.Identity()
    with pytest.raises(ValueError, match="activation checkpointing"):
        enc.set_grad_checkpointing(True)


def test_builder_applies_both_options(tmp_path, monkeypatch):
    from src.training.builder import TrainingSessionBuilder
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(grad_accum_steps=3, grad_checkpointing=True)
    cfg["model"] = {"name": "vit_tiny_patch16_224", "pretrained": False, "img_size": 32}
    cfg["checkpoint"] = {"dir": str(tmp_path / "ckpt")}
    trainer = TrainingSessionBuilder(cfg, torch.device("cpu")).with_trace("off").build()
    assert trainer.grad_accum_steps == 3
    assert trainer.model.backbone.grad_checkpointing


# ── Benchmark ────────────────────────────────────────────────────────────────

def test_benchmarker_steps_the_optimizer_every_k_microbatches():
    from src.benchmark.benchmarker import Benchmarker
    model = TimmClassifier("resnet18", num_classes=N_CLASSES, pretrained=False)
    bench = Benchmarker(model, torch.device("cpu"), num_classes=N_CLASSES, grad_accum_steps=4)
    steps = []
    bench._optimizer.step = lambda *a, **k: steps.append(1)
    images, labels = torch.randn(2, 3, 32, 32), torch.zeros(2, N_CLASSES)
    model.train()
    for _ in range(8):
        bench._train_step(images, labels)
    assert len(steps) == 2
