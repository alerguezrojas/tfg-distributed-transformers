"""Unit tests for src/models/model_parallel.py.

The key correctness risk is that the model-parallel forward re-implements timm's
forward_features + forward_head. These tests pin it to the stock model's output
on CPU (both "stages" on cpu), so the only thing that changes on real hardware
is tensor placement, not the math. No GPU required.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.model_parallel import ModelParallelViT, build_model_parallel_vit
from src.models.vit import build_model

MODEL = "vit_tiny_patch16_224"


@pytest.fixture(scope="module")
def base_model():
    # pretrained=False keeps the test offline and fast.
    return build_model(MODEL, num_classes=19, pretrained=False).eval()


def test_forward_matches_stock_model_on_cpu(base_model):
    """Model-parallel forward must equal the stock forward (same weights, CPU)."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = base_model(x)
    mp = ModelParallelViT(base_model, devices=["cpu", "cpu"], split_block=6).eval()
    with torch.no_grad():
        out = mp(x)
    assert out.shape == ref.shape == (2, 19)
    assert torch.allclose(ref, out, atol=1e-5), (ref - out).abs().max().item()


@pytest.mark.parametrize("split", [1, 3, 6, 9, 11])
def test_equivalence_for_various_split_points(base_model, split):
    torch.manual_seed(1)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = base_model(x)
        mp = ModelParallelViT(base_model, devices=["cpu", "cpu"], split_block=split).eval()
        out = mp(x)
    assert torch.allclose(ref, out, atol=1e-5)


def test_invalid_split_raises(base_model):
    n = len(base_model.backbone.blocks)
    with pytest.raises(ValueError):
        ModelParallelViT(base_model, devices=["cpu", "cpu"], split_block=0)
    with pytest.raises(ValueError):
        ModelParallelViT(base_model, devices=["cpu", "cpu"], split_block=n)


def test_non_vit_rejected():
    cnn = build_model("resnet50", num_classes=19, pretrained=False)
    with pytest.raises(ValueError):
        ModelParallelViT(cnn, devices=["cpu", "cpu"])


def test_output_device_is_last_stage(base_model):
    mp = ModelParallelViT(base_model, devices=["cpu", "cpu"], split_block=6)
    assert mp.output_device == torch.device("cpu")


def test_gradient_flows_across_split(base_model):
    """Backward must propagate through the device boundary to stage-0 params."""
    mp = ModelParallelViT(base_model, devices=["cpu", "cpu"], split_block=6)
    x = torch.randn(2, 3, 224, 224)
    target = torch.zeros(2, 19)
    out = mp(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
    loss.backward()
    # A stage-0 parameter (patch_embed) must have received a gradient.
    g = mp.base.backbone.patch_embed.proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_builder_helper():
    mp = build_model_parallel_vit(MODEL, pretrained=False, devices=["cpu", "cpu"], split_block=4)
    assert isinstance(mp, ModelParallelViT)
    assert mp.split_block == 4
