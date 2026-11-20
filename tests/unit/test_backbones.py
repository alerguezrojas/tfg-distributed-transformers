"""Backbone registry, name resolution, the ``hf:`` adapter (with a fake model: no
network, no ``transformers``), linear probe (freeze_backbone) and its validation."""
import sys
import types

import pytest
import torch
import torch.nn as nn

from src.models import backbones, vit
from src.models.backbones import (
    BackboneSpec, available_backbones, register_backbone, resolve_backbone,
)
from src.models.vit import TimmClassifier, build_llrd_optimizer
from src.training.builder import TrainingSessionBuilder
from src.training.config_validator import ConfigError, validate_config


# ── A tiny stand-in for a transformers ViT (I-JEPA-like: embeddings + encoder.layer) ──

class _Out:
    def __init__(self, h):
        self.last_hidden_state = h


class _FakeHFModel(nn.Module):
    def __init__(self, hidden=16, layers=3, image_size=32, patch=16):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden, image_size=image_size)
        self.embeddings = nn.Conv2d(3, hidden, patch, patch)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.layernorm = nn.LayerNorm(hidden)
        self.seen_kwargs = None

    def forward(self, pixel_values, **kwargs):
        self.seen_kwargs = kwargs
        t = self.embeddings(pixel_values).flatten(2).transpose(1, 2)    # (B, N, D)
        for blk in self.encoder.layer:
            t = blk(t)
        return _Out(self.layernorm(t))


@pytest.fixture
def fake_transformers(monkeypatch):
    """A ``transformers`` module whose AutoModel returns the fake model."""
    mod = types.ModuleType("transformers")
    mod.calls = []

    class AutoModel:
        @staticmethod
        def from_pretrained(repo):
            mod.calls.append(("pretrained", repo))
            return _FakeHFModel()

        @staticmethod
        def from_config(cfg):
            mod.calls.append(("config", cfg))
            return _FakeHFModel()

    class AutoConfig:
        @staticmethod
        def from_pretrained(repo):
            return repo

    mod.AutoModel, mod.AutoConfig = AutoModel, AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return mod


@pytest.fixture(autouse=True)
def _reset_warnings():
    vit._warned.clear()
    yield
    vit._warned.clear()


# ── Registry and resolution ──────────────────────────────────────────────────

class TestRegistry:
    def test_ijepa_registered_with_license(self):
        for name in ("ijepa_vith14_1k", "ijepa_vith14_22k"):
            spec = resolve_backbone(name)
            assert spec.source == "transformers" and spec.pooling == "mean"
            assert spec.license == "CC BY-NC 4.0" and spec.non_commercial
        assert {"ijepa_vith14_1k", "ijepa_vith14_22k"} <= set(available_backbones())

    def test_timm_names_resolve_to_none(self):
        assert resolve_backbone("vit_base_patch16_224") is None
        assert resolve_backbone("resnet18") is None

    def test_hf_prefix_builds_generic_spec(self):
        spec = resolve_backbone("hf:facebook/dinov2-small")
        assert spec.source == "transformers" and spec.pooling == "mean" and spec.llrd
        with pytest.raises(ValueError, match="needs a repository"):
            resolve_backbone("hf:")

    def test_pooling_override(self):
        assert resolve_backbone("ijepa_vith14_1k", "cls").pooling == "cls"
        assert resolve_backbone("hf:a/b", "cls").pooling == "cls"

    def test_register_duplicate_and_validation(self):
        spec = BackboneSpec(name="toy_dup", source="torch_hub", build=lambda cfg: nn.Identity())
        register_backbone(spec)
        try:
            assert register_backbone(spec) is spec          # idempotent
            other = BackboneSpec(name="toy_dup", source="torch_hub", build=lambda cfg: None)
            with pytest.raises(ValueError, match="already registered"):
                register_backbone(other)
        finally:
            backbones._REGISTRY.pop("toy_dup", None)
        with pytest.raises(ValueError, match="source"):
            BackboneSpec(name="x", source="nope", build=lambda cfg: None)
        with pytest.raises(ValueError, match="pooling"):
            BackboneSpec(name="x", source="timm", build=lambda cfg: None, pooling="max")
        with pytest.raises(ValueError, match="lowercase"):
            BackboneSpec(name="X", source="timm", build=lambda cfg: None)


# ── timm unchanged ───────────────────────────────────────────────────────────

class TestTimmUnchanged:
    def test_timm_model_has_no_spec_and_same_defaults(self):
        m = TimmClassifier("resnet18", num_classes=3, pretrained=False)
        assert m.backbone_spec is None and m.img_size == vit.DEFAULT_IMG_SIZE
        assert not m.is_vit and all(p.requires_grad for p in m.parameters())
        assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 3)

    def test_timm_vit_llrd_groups_cover_all_params(self):
        m = TimmClassifier("vit_tiny_patch16_224", num_classes=3, pretrained=False)
        opt = build_llrd_optimizer(m, 1e-3, 0.05, 0.75)
        n_in_groups = sum(len(g["params"]) for g in opt.param_groups)
        assert n_in_groups == len(list(m.parameters()))
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)       # head first
        assert min(g["lr"] for g in opt.param_groups) < 1e-3 * 0.75 ** 12


# ── hf: adapter ──────────────────────────────────────────────────────────────

class TestHFAdapter:
    def test_missing_transformers_is_a_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", None)    # makes the import fail
        with pytest.raises(ImportError, match=r"optional 'transformers'.*\[hf\]"):
            TimmClassifier("hf:facebook/ijepa_vith14_1k", num_classes=2, pretrained=False)

    def test_forward_mean_and_cls_pooling(self, fake_transformers):
        mean = TimmClassifier("hf:org/m", num_classes=4, pretrained=True)
        assert fake_transformers.calls[0] == ("pretrained", "org/m")
        x = torch.randn(2, 3, 32, 32)
        assert mean(x).shape == (2, 4) and mean.backbone.num_features == 16
        feats = mean.backbone(x)
        tokens = mean.backbone.model(x).last_hidden_state
        assert torch.allclose(feats, tokens.mean(dim=1))
        cls = TimmClassifier("hf:org/m", num_classes=4, pretrained=False, pooling="cls")
        assert fake_transformers.calls[1][0] == "config"
        cls.backbone.model.load_state_dict(mean.backbone.model.state_dict())
        assert torch.allclose(cls.backbone(x), tokens[:, 0])

    def test_other_input_side_interpolates_position_embeddings(self, fake_transformers):
        m = TimmClassifier("hf:org/m", num_classes=2, pretrained=False)
        m(torch.zeros(1, 3, 32, 32))
        assert m.backbone.model.seen_kwargs == {}
        m(torch.zeros(1, 3, 48, 48))
        assert m.backbone.model.seen_kwargs == {"interpolate_pos_encoding": True}

    def test_registered_ijepa_uses_its_repo(self, fake_transformers):
        m = TimmClassifier("ijepa_vith14_1k", num_classes=2, pretrained=True)
        assert fake_transformers.calls == [("pretrained", "facebook/ijepa_vith14_1k")]
        assert m.img_size == 224 and m.is_vit

    def test_needs_three_channels(self, fake_transformers):
        with pytest.raises(ValueError, match="3-channel"):
            TimmClassifier("hf:org/m", num_classes=2, in_chans=1)

    def test_llrd_layout_groups(self, fake_transformers):
        m = TimmClassifier("hf:org/m", num_classes=2, pretrained=False)
        opt = build_llrd_optimizer(m, 1e-3, 0.05, 0.75)
        assert sum(len(g["params"]) for g in opt.param_groups) == len(list(m.parameters()))
        lrs = sorted({g["lr"] for g in opt.param_groups})
        assert len(lrs) == 5       # head, top = last block, 2 more blocks, embeddings
        # no weight decay on the (lowercased) norm and bias parameters
        assert any(g["weight_decay"] == 0.0 for g in opt.param_groups)


# ── Warnings ─────────────────────────────────────────────────────────────────

class TestWarnings:
    def test_non_commercial_license_warned_once(self, fake_transformers, capsys):
        TimmClassifier("ijepa_vith14_1k", num_classes=2, pretrained=False)
        TimmClassifier("ijepa_vith14_1k", num_classes=2, pretrained=False)
        out = capsys.readouterr().out
        assert out.count("CC BY-NC 4.0") == 1 and "no comercial" in out

    def test_hf_generic_has_no_license_warning(self, fake_transformers, capsys):
        TimmClassifier("hf:org/m", num_classes=2, pretrained=False)
        assert "[aviso]" not in capsys.readouterr().out

    def test_other_normalization_warned(self, capsys):
        spec = BackboneSpec(name="toy_norm", source="torch_hub", mean=(0.5,) * 3, std=(0.5,) * 3,
                            build=lambda cfg: _Feat())
        register_backbone(spec)
        try:
            TimmClassifier("toy_norm", num_classes=2, pretrained=False)
        finally:
            backbones._REGISTRY.pop("toy_norm", None)
        assert "normalizan con la media/desviación de ImageNet" in capsys.readouterr().out


class _Feat(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 8)

    def forward(self, x):
        return self.lin(x.mean(dim=(2, 3)))


# ── Linear probe ─────────────────────────────────────────────────────────────

class TestFreezeBackbone:
    def test_only_head_gets_gradient(self):
        m = TimmClassifier("resnet18", num_classes=3, pretrained=False, freeze_backbone=True)
        m.train()
        assert not m.backbone.training                      # frozen features stay in eval
        m(torch.randn(2, 3, 64, 64)).sum().backward()
        assert all(p.grad is None for p in m.backbone.parameters())
        assert all(p.grad is not None for p in m.head.parameters())

    def test_not_frozen_by_default(self):
        m = TimmClassifier("resnet18", num_classes=3, pretrained=False)
        assert all(p.requires_grad for p in m.backbone.parameters())

    def _cfg(self, tmp_path, **model):
        return {
            "data": {"root": "/r", "metadata": "/m.parquet"},
            "model": {"name": "vit_tiny_patch16_224", "pretrained": False, **model},
            "training": {"epochs": 1, "batch_size": 4, "lr": 0.001, "llrd_decay": 0.75},
            "checkpoint": {"dir": str(tmp_path / "ckpt")},
        }

    def test_builder_optimizer_only_sees_the_head(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = self._cfg(tmp_path, freeze_backbone=True)
        trainer = TrainingSessionBuilder(cfg, torch.device("cpu")).with_trace("off").build()
        in_opt = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
        assert in_opt == {id(p) for p in trainer.model.head.parameters()}

    def test_hf_backbone_through_the_builder(self, tmp_path, monkeypatch, fake_transformers):
        monkeypatch.chdir(tmp_path)
        cfg = self._cfg(tmp_path)
        cfg["model"].update(name="hf:org/m", pooling="cls", img_size=32)
        trainer = TrainingSessionBuilder(cfg, torch.device("cpu")).with_trace("off").build()
        assert trainer.model.backbone.pooling == "cls"
        assert trainer.model.backbone_spec.name == "hf:org/m"


# ── Validator ────────────────────────────────────────────────────────────────

def _cfg(**model):
    return {"data": {"root": "/r", "metadata": "/m.parquet"},
            "model": {"name": "vit_tiny_patch16_224", **model},
            "training": {"epochs": 1, "batch_size": 4, "lr": 0.001}}


class TestValidator:
    def test_valid_configs(self):
        validate_config(_cfg())
        validate_config(_cfg(name="hf:facebook/ijepa_vith14_1k", pooling="mean", freeze_backbone=True))
        validate_config(_cfg(name="ijepa_vith14_1k"))
        validate_config(_cfg(freeze_backbone=True))

    def test_bad_pooling_and_freeze(self):
        with pytest.raises(ConfigError, match="model.pooling must be one of"):
            validate_config(_cfg(name="hf:a/b", pooling="max"))
        with pytest.raises(ConfigError, match="freeze_backbone must be true or false"):
            validate_config(_cfg(freeze_backbone="yes"))

    def test_pooling_on_timm_model_rejected(self):
        with pytest.raises(ConfigError, match="only applies to registered or 'hf:'"):
            validate_config(_cfg(pooling="cls"))

    def test_empty_hf_repo(self):
        with pytest.raises(ConfigError, match="needs a repository"):
            validate_config(_cfg(name="hf:"))

    def test_non_timm_with_model_parallel_rejected(self):
        validate_config(_cfg(), model_parallel=True)
        with pytest.raises(ConfigError, match="model parallelism"):
            validate_config(_cfg(name="hf:a/b"), model_parallel=True)
        with pytest.raises(ConfigError, match="freeze_backbone is not supported"):
            validate_config(_cfg(freeze_backbone=True), model_parallel=True)

    def test_non_timm_needs_three_channel_dataset(self):
        cfg = _cfg(name="hf:a/b")
        cfg["data"]["name"] = "medmnist_breast"          # 1 channel
        cfg["data"].update(root="/r", size=28, download=False)
        with pytest.raises(ConfigError, match="expects 3-channel"):
            validate_config(cfg)

    def test_model_parallel_builder_refuses_non_timm(self):
        from src.models.model_parallel import build_model_parallel_vit
        with pytest.raises(ValueError, match="only splits timm"):
            build_model_parallel_vit("hf:a/b", devices=["cpu", "cpu"])
