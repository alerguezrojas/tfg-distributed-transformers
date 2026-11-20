"""The ``hf:`` adapter against the REAL ``transformers`` library (tiny I-JEPA built from a
config: no downloads). Skipped when the optional dependency is not installed."""
import pytest
import torch

pytest.importorskip("transformers")
from transformers import IJepaConfig, IJepaModel  # noqa: E402

from src.models.backbones import HFEncoder, build_hf_encoder  # noqa: E402
from src.models.vit import TimmClassifier, build_llrd_optimizer  # noqa: E402

HIDDEN, LAYERS, SIDE, PATCH = 32, 3, 32, 16


def _config():
    return IJepaConfig(hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=2,
                       intermediate_size=64, image_size=SIDE, patch_size=PATCH)


@pytest.fixture
def encoder():
    return HFEncoder(IJepaModel(_config(), add_pooling_layer=False))


def test_build_from_config_without_network():
    enc = build_hf_encoder("facebook/ijepa_vith14_1k", {"pretrained": False, "in_chans": 3},
                           config=_config())
    assert isinstance(enc, HFEncoder) and enc.num_features == HIDDEN


def test_shape_and_pooling(encoder):
    x = torch.randn(2, 3, SIDE, SIDE)
    feats = encoder(x)
    assert feats.shape == (2, HIDDEN)
    tokens = encoder.model(pixel_values=x).last_hidden_state
    assert torch.allclose(feats, tokens.mean(dim=1))
    encoder.pooling = "cls"
    assert torch.allclose(encoder(x), tokens[:, 0])


def test_other_resolution_interpolates_positions(encoder):
    assert encoder(torch.randn(2, 3, 48, 48)).shape == (2, HIDDEN)


def test_llrd_layout_uses_real_parameter_names(encoder):
    embed, blocks, rest = encoder.llrd_layout()
    assert len(blocks) == LAYERS
    assert all(n.startswith(("position_embeddings", "patch_embeddings")) for n, _ in embed)
    assert any(n.startswith("attention.q_proj") for n, _ in blocks[0])   # names relative to the block
    assert any("layernorm" in n for n, _ in rest)
    total = sum(len(g) for g in (embed, rest, *blocks))
    assert total == len(list(encoder.parameters()))


def test_classifier_llrd_and_freeze(encoder, monkeypatch):
    monkeypatch.setattr("src.models.backbones.build_hf_encoder",
                        lambda repo, cfg, config=None: encoder)
    model = TimmClassifier("hf:org/tiny", num_classes=3, pretrained=False)
    opt = build_llrd_optimizer(model, 1e-3, 0.05, 0.75)
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))

    frozen = TimmClassifier("hf:org/tiny", num_classes=3, pretrained=False, freeze_backbone=True)
    frozen.train()
    frozen(torch.randn(2, 3, SIDE, SIDE)).sum().backward()
    assert all(p.grad is None for p in frozen.backbone.parameters())
    assert all(p.grad is not None for p in frozen.head.parameters())
