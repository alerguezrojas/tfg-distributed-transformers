"""Naive model (pipeline) parallelism for the BigEarthNet ViT.

Splits a timm Vision Transformer across two devices: the patch embedding and the
first ``split_block`` transformer blocks live on ``devices[0]``; the remaining
blocks, the final norm, the pooling/head and the custom multi-label head live on
``devices[1]``. In the forward pass the activation is copied across the device
boundary once (``x.to(devices[1])``); autograd carries the gradient back across
it automatically.

This is the textbook *naive* model-parallel baseline: it lets a model that does
not fit on a single GPU run across several, but it does **not** overlap the
stages (while stage 1 computes, stage 0 is idle), so on a model that already
fits in one GPU it is slower than data parallelism. Pipelining micro-batches
(GPipe) would recover utilisation — noted as future work.

The forward is a faithful re-implementation of timm's
``forward_features`` + ``forward_head``; ``tests/unit/test_model_parallel.py``
asserts it matches the stock model numerically (on CPU), so the only behaviour
that differs on real hardware is *where* each tensor lives.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.vit import BigEarthModel


class ModelParallelViT(nn.Module):
    """Wrap a :class:`BigEarthModel` (ViT backbone) split across two devices."""

    def __init__(
        self,
        base: BigEarthModel,
        devices: list[str] | None = None,
        split_block: int | None = None,
    ):
        super().__init__()
        if not getattr(base, "is_vit", False):
            raise ValueError(
                f"ModelParallelViT requires a ViT/DeiT/Swin backbone, got "
                f"'{base.model_name}'."
            )
        self.base = base
        devices = devices or ["cuda:0", "cuda:1"]
        self.dev0 = torch.device(devices[0])
        self.dev1 = torch.device(devices[-1])

        n_blocks = len(base.backbone.blocks)
        if split_block is None:
            split_block = n_blocks // 2
        if not (0 < split_block < n_blocks):
            raise ValueError(
                f"split_block must be in (0, {n_blocks}); got {split_block}."
            )
        self.split_block = split_block
        self._place()

    # ── device placement ─────────────────────────────────────────────────────
    def _place(self) -> None:
        bb = self.base.backbone

        # Stage 0 → dev0: patch embed, prefix params, pre-norm, first blocks.
        for name in ("patch_embed", "pos_drop", "patch_drop", "norm_pre"):
            mod = getattr(bb, name, None)
            if isinstance(mod, nn.Module):
                mod.to(self.dev0)
        for pname in ("cls_token", "pos_embed", "reg_token"):
            p = getattr(bb, pname, None)
            if isinstance(p, nn.Parameter):
                p.data = p.data.to(self.dev0)
        for blk in bb.blocks[: self.split_block]:
            blk.to(self.dev0)

        # Stage 1 → dev1: remaining blocks, final norm, pooling/head.
        for blk in bb.blocks[self.split_block:]:
            blk.to(self.dev1)
        for name in ("norm", "fc_norm", "head_drop", "head", "attn_pool"):
            mod = getattr(bb, name, None)
            if isinstance(mod, nn.Module):
                mod.to(self.dev1)
        self.base.head.to(self.dev1)

    # ── forward (mirrors timm forward_features + forward_head) ────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb = self.base.backbone
        x = x.to(self.dev0)
        x = bb.patch_embed(x)
        x = bb._pos_embed(x)            # adds cls token + pos embed (+ pos_drop)
        if hasattr(bb, "patch_drop"):
            x = bb.patch_drop(x)
        if hasattr(bb, "norm_pre"):
            x = bb.norm_pre(x)
        for i, blk in enumerate(bb.blocks):
            if i == self.split_block:
                x = x.to(self.dev1)
            x = blk(x)
        x = bb.norm(x)
        x = bb.forward_head(x)          # pooling + fc_norm + head (Identity, num_classes=0)
        return self.base.head(x)

    @property
    def output_device(self) -> torch.device:
        """Device the logits live on — move targets/loss here."""
        return self.dev1


def build_model_parallel_vit(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 19,
    pretrained: bool = True,
    devices: list[str] | None = None,
    split_block: int | None = None,
) -> ModelParallelViT:
    """Build a ViT and wrap it for model parallelism across ``devices``."""
    base = BigEarthModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    return ModelParallelViT(base, devices=devices, split_block=split_block)
