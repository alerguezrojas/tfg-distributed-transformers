"""OUSM (``training.ousm_k``): the k highest-loss samples of each training batch are ignored."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.training.losses import (
    CrossEntropyEMDLoss,
    FocalLoss,
    OUSMLoss,
    build_criterion,
    squared_emd,
)
from src.training.tasks import get_task

N, C = 8, 6


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(N, C, generator=g)
    targets = F.one_hot(torch.randint(0, C, (N,), generator=g), C).float()
    return logits, targets


def _bases():
    return {
        "ce": nn.CrossEntropyLoss(),
        "ce_emd": CrossEntropyEMDLoss(emd_weight=0.5),
        "bce_pos_weight": nn.BCEWithLogitsLoss(pos_weight=torch.full((C,), 2.0)),
        "focal": FocalLoss(gamma=2.0),
    }


class TestOUSMLoss:
    @pytest.mark.parametrize("name", list(_bases()))
    def test_without_grad_it_is_the_plain_mean(self, name):
        """Validation (no_grad) sees every sample, so its loss matches a run without OUSM."""
        logits, targets = _batch()
        plain = _bases()[name](logits, targets)
        with torch.no_grad():
            mined = OUSMLoss(_bases()[name], k=3)(logits, targets)
        assert torch.allclose(mined, plain, atol=1e-6)

    @pytest.mark.parametrize("name", list(_bases()))
    def test_k_zero_is_the_plain_mean_while_training(self, name):
        logits, targets = _batch(1)
        plain = _bases()[name](logits, targets)
        assert torch.allclose(OUSMLoss(_bases()[name], k=0)(logits, targets), plain, atol=1e-6)

    def test_drops_the_k_highest_losses(self):
        logits, targets = _batch(2)
        per_sample = F.cross_entropy(logits, targets, reduction="none")
        expected = per_sample.sort().values[: N - 2].mean()
        got = OUSMLoss(nn.CrossEntropyLoss(), k=2)(logits.requires_grad_(), targets)
        assert torch.allclose(got, expected)

    def test_dropped_samples_get_no_gradient(self):
        logits, targets = _batch(3)
        logits.requires_grad_()
        worst = F.cross_entropy(logits, targets, reduction="none").argmax()
        OUSMLoss(nn.CrossEntropyLoss(), k=1)(logits, targets).backward()
        assert torch.all(logits.grad[worst] == 0)
        others = [i for i in range(N) if i != worst]
        assert logits.grad[others].abs().sum() > 0

    def test_keeps_at_least_one_sample(self):
        logits, targets = _batch(4)
        logits, targets = logits[:2].requires_grad_(), targets[:2]
        per_sample = F.cross_entropy(logits, targets, reduction="none")
        got = OUSMLoss(nn.CrossEntropyLoss(), k=5)(logits, targets)
        assert torch.allclose(got, per_sample.min())

    def test_ce_emd_per_sample_matches_its_mean(self):
        logits, targets = _batch(5)
        loss = CrossEntropyEMDLoss(emd_weight=0.5)
        mean = loss(logits, targets)
        loss.reduction = "none"
        per_sample = loss(logits, targets)
        assert per_sample.shape == (N,)
        assert torch.allclose(per_sample.mean(), mean, atol=1e-6)
        expected = F.cross_entropy(logits, targets, reduction="none") + 0.5 * squared_emd(logits, targets)
        assert torch.allclose(per_sample, expected)

    def test_exposes_the_wrapped_pos_weight(self):
        pw = torch.full((C,), 3.0)
        assert OUSMLoss(nn.BCEWithLogitsLoss(pos_weight=pw), k=1).pos_weight is pw
        assert OUSMLoss(nn.CrossEntropyLoss(), k=1).pos_weight is None

    @pytest.mark.parametrize("bad", [-1, 1.5, "2", True])
    def test_bad_k_rejected(self, bad):
        with pytest.raises(ValueError, match="ousm_k"):
            OUSMLoss(nn.CrossEntropyLoss(), k=bad)

    def test_loss_without_reduction_rejected(self):
        class NoReduction(nn.Module):
            def forward(self, logits, targets):
                return logits.sum()

        with pytest.raises(ValueError, match="reduction"):
            OUSMLoss(NoReduction(), k=1)


class TestFactory:
    def test_unset_or_zero_gives_the_plain_loss(self):
        task = get_task("multiclass", ordinal=True)
        assert isinstance(build_criterion({}, task=task), nn.CrossEntropyLoss)
        assert isinstance(build_criterion({"ousm_k": 0}, task=task), nn.CrossEntropyLoss)

    def test_wraps_the_configured_loss(self):
        task = get_task("multiclass", ordinal=True)
        crit = build_criterion({"loss": "ce_emd", "emd_weight": 0.5, "ousm_k": 2}, task=task)
        assert isinstance(crit, OUSMLoss) and crit.k == 2
        assert isinstance(crit.base, CrossEntropyEMDLoss) and crit.base.emd_weight == pytest.approx(0.5)

    def test_heterogeneous_builder_rejects_it(self):
        from src.data.registry import get_dataset_spec
        from src.training.builder import TrainingSessionBuilder
        cfg = {"data": {"root": ".", "metadata": "x.parquet"},
               "model": {"name": "vit_tiny_patch16_224", "num_classes": 19},
               "training": {"epochs": 1, "batch_size": 4, "lr": 0.001, "ousm_k": 1},
               "checkpoint": {"dir": "checkpoints"}}
        builder = TrainingSessionBuilder(cfg, torch.device("cpu"))
        builder._hetero_local_batch_size = 4
        with pytest.raises(ValueError, match="heterogeneous"):
            builder._build_criterion(cfg, get_dataset_spec("bigearthnet"))


class TestValidator:
    @staticmethod
    def _cfg(**train):
        return {"data": {"root": "/tmp/ds", "metadata": "/tmp/meta.parquet"},
                "model": {"name": "vit_tiny_patch16_224", "num_classes": 19},
                "training": {"epochs": 10, "batch_size": 16, "lr": 0.0001, **train}}

    def test_valid(self):
        from src.training.config_validator import validate_config
        validate_config(self._cfg(ousm_k=2))
        validate_config(self._cfg(ousm_k=0))

    @pytest.mark.parametrize("bad", [-1, 1.5, "2", True, 16])
    def test_invalid(self, bad):
        from src.training.config_validator import ConfigError, validate_config
        with pytest.raises(ConfigError, match="ousm_k"):
            validate_config(self._cfg(ousm_k=bad))


def test_trainer_mines_only_in_training(tmp_path):
    """In the Trainer, the training loss drops samples and the validation loss does not."""
    from src.training.trainer import Trainer
    torch.manual_seed(0)
    x = torch.randn(16, 3, 8, 8)
    y = F.one_hot(torch.randint(0, C, (16,)), C).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, C))
    task = get_task("multiclass", ordinal=True)

    def trainer(criterion):
        opt = torch.optim.SGD(model.parameters(), lr=0.0)      # lr 0: weights never move
        return Trainer(model, opt, None, torch.device("cpu"), checkpoint_dir=str(tmp_path),
                       criterion=criterion, task=task)

    plain = trainer(nn.CrossEntropyLoss())
    mined = trainer(OUSMLoss(nn.CrossEntropyLoss(), k=3))
    assert mined.eval_epoch(loader)["loss"] == pytest.approx(plain.eval_epoch(loader)["loss"])
    assert mined.train_epoch(loader)["loss"] < plain.train_epoch(loader)["loss"]


def test_log_field_and_its_parsers():
    """``ousm=k`` in the ``Configuración:`` line, read back by the backfill and the web."""
    from src.run_manifest import _training_from_config_line
    from src.training.losses import ousm_log_field
    assert ousm_log_field({"ousm_k": 2}) == " | ousm=2"
    assert ousm_log_field({}) == "" and ousm_log_field({"ousm_k": 0}) == ""
    assert _training_from_config_line({"ousm": "2", "seed": "43"}, None)["ousm_k"] == 2
    assert "ousm_k" not in _training_from_config_line({"seed": "43"}, None)


def test_web_reads_ousm_from_the_log_when_there_is_no_manifest(tmp_path):
    from src.web.run_meta import resolve_meta
    log = tmp_path / "train_21112026_061525.log"
    log.write_text("2026-11-21 06:15:28 [INFO ] Configuración: modelo=vit_small_patch16_224 | "
                   "batch=16/GPU (global=16) | loss=ce | dataset=panda | seed=42 | "
                   "backbone=timm | ousm=2\n", encoding="utf-8")
    meta = resolve_meta(manifest=None, run_start=None, log_path=log, dataset="panda")
    assert meta.ousm_k == 2 and "OUSM 2" in meta.tags()
