"""LLRD for any timm backbone (``group_matcher``), not only ViT."""
import pytest
import torch.nn as nn

from src.models.vit import (
    TimmClassifier,
    _group_matcher_layout,
    _llrd_layout,
    build_llrd_optimizer,
)

LR, WD, DECAY = 1e-3, 0.05, 0.75
OTHERS = ["resnet18", "efficientnet_b0", "convnext_atto", "swin_tiny_patch4_window7_224"]


def _names(groups):
    return [sorted(n for n, _ in g) for g in groups]


def _lr_of(opt):
    return {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}


def test_vit_generic_layout_matches_the_vit_one():
    """On a ViT, timm's group_matcher gives the same 12 blocks as the blocks/patch_embed
    path (the final norm folded into the last one), so both paths mean the same thing."""
    bb = TimmClassifier("vit_tiny_patch16_224", num_classes=3, pretrained=False).backbone
    _, blocks, _ = _llrd_layout(bb)
    g_embed, g_blocks, g_top = _group_matcher_layout(bb)
    assert g_top == [] and len(g_blocks) == len(blocks) == 12
    vit = [sorted(f"blocks.{i}.{n}" for n, _ in blk) for i, blk in enumerate(blocks)]
    gen = _names(g_blocks)
    assert gen[:-1] == vit[:-1]
    assert set(gen[-1]) - set(vit[-1]) == {"norm.weight", "norm.bias"}
    assert {n for n, _ in g_embed} >= {"patch_embed.proj.weight", "cls_token", "pos_embed"}


@pytest.mark.parametrize("name", OTHERS)
def test_other_families_get_llrd(name):
    m = TimmClassifier(name, num_classes=4, pretrained=False)
    assert m.supports_llrd
    opt = build_llrd_optimizer(m, LR, WD, DECAY)
    params = [p for p in m.parameters() if p.requires_grad]
    in_groups = [p for g in opt.param_groups for p in g["params"]]
    assert len(in_groups) == len(params) and {id(p) for p in in_groups} == {id(p) for p in params}
    lr = _lr_of(opt)
    assert all(lr[id(p)] == pytest.approx(LR) for p in m.head.parameters())
    embed, blocks, _ = _group_matcher_layout(m.backbone)
    assert len(blocks) >= 4
    stem_lr = {lr[id(p)] for _, p in embed}
    assert stem_lr == {min(lr.values())}
    block_lrs = [lr[id(blocks[i][0][1])] for i in range(len(blocks))]
    assert block_lrs == sorted(block_lrs) and block_lrs[0] < block_lrs[-1]   # deeper = higher


def test_swin_no_longer_breaks():
    """swin_ was listed as ViT but has no ``.blocks``: LLRD used to raise AttributeError."""
    m = TimmClassifier("swin_tiny_patch4_window7_224", num_classes=2, pretrained=False)
    assert not hasattr(m.backbone, "blocks")
    assert len(build_llrd_optimizer(m, LR, WD, DECAY).param_groups) > 4


def test_vit_optimizer_unchanged():
    """The ViT path (golden-tested behaviour) is the same as before the change."""
    m = TimmClassifier("vit_tiny_patch16_224", num_classes=3, pretrained=False)
    opt = build_llrd_optimizer(m, LR, WD, DECAY)
    lr = _lr_of(opt)
    assert lr[id(m.backbone.patch_embed.proj.weight)] == pytest.approx(LR * DECAY ** 14)
    assert lr[id(m.backbone.blocks[-1].attn.qkv.weight)] == pytest.approx(LR * DECAY)
    assert lr[id(m.backbone.blocks[0].attn.qkv.weight)] == pytest.approx(LR * DECAY ** 12)


class _Plain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)


def test_backbone_without_layers_is_a_clear_error():
    m = TimmClassifier("resnet18", num_classes=2, pretrained=False)
    m.backbone = _Plain()
    assert not m.supports_llrd
    with pytest.raises(ValueError, match="llrd_decay: 0.0"):
        build_llrd_optimizer(m, LR, WD, DECAY)

