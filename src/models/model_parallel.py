"""Naive model (pipeline) parallelism for the BigEarthNet ViT.

Splits a timm Vision Transformer across two devices: the patch embedding and the
first ``split_block`` transformer blocks live on ``devices[0]``; the remaining
blocks, the final norm, the pooling/head and the custom multi-label head live on
``devices[1]``. In the forward pass the activation is copied across the device
boundary once (``x.to(devices[1])``); autograd carries the gradient back across
it automatically.

With ``micro_batches=1`` this is the textbook *naive* model-parallel baseline: it
lets a model that does not fit on a single GPU run across several, but it does
**not** overlap the stages (while stage 1 computes, stage 0 is idle). With
``micro_batches=m > 1`` the batch is split into ``m`` micro-batches and pipelined
GPipe-style (Huang et al., 2019): while stage 1 runs micro-batch ``i``, stage 0
already runs micro-batch ``i+1`` (CUDA launches are asynchronous, so one Python
thread keeps both devices busy). The step stays synchronous — the gradients of the
``m`` micro-batches accumulate before the optimizer step — so the update is the same
as with the whole batch (the model has no BatchNorm: LayerNorm is per sample).

The forward is a faithful re-implementation of timm's
``forward_features`` + ``forward_head``; ``tests/unit/test_model_parallel.py``
asserts it matches the stock model numerically (on CPU), so the only behaviour
that differs on real hardware is *where* each tensor lives.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones import resolve_backbone
from src.models.vit import TimmClassifier


class ModelParallelViT(nn.Module):
    """Wrap a :class:`TimmClassifier` (ViT backbone) split across two devices."""

    def __init__(
        self,
        base: TimmClassifier,
        devices: list[str] | None = None,
        split_block: int | None = None,
        micro_batches: int = 1,
    ):
        super().__init__()
        if micro_batches < 1:
            raise ValueError(f"micro_batches must be >= 1; got {micro_batches}.")
        self.micro_batches = int(micro_batches)
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
    def _stage0(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embedding and the first ``split_block`` blocks, on ``dev0``; the result
        is handed to ``dev1`` (the one copy across the device boundary)."""
        bb = self.base.backbone
        x = x.to(self.dev0)
        x = bb.patch_embed(x)
        x = bb._pos_embed(x)            # adds cls token + pos embed (+ pos_drop)
        if hasattr(bb, "patch_drop"):
            x = bb.patch_drop(x)
        if hasattr(bb, "norm_pre"):
            x = bb.norm_pre(x)
        for blk in bb.blocks[: self.split_block]:
            x = blk(x)
        return x.to(self.dev1)

    def _stage1(self, x: torch.Tensor) -> torch.Tensor:
        """The remaining blocks, the final norm and the heads, on ``dev1``."""
        bb = self.base.backbone
        for blk in bb.blocks[self.split_block:]:
            x = blk(x)
        x = bb.norm(x)
        x = bb.forward_head(x)          # pooling + fc_norm + head (Identity, num_classes=0)
        return self.base.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_micro = min(self.micro_batches, x.shape[0])
        if n_micro <= 1:
            return self._stage1(self._stage0(x))
        # GPipe forward: stage 0 of micro-batch i+1 is launched before stage 1 of micro-batch i
        # finishes, so the two devices overlap; autograd keeps the graph of every micro-batch
        # and one backward over the concatenated logits accumulates all their gradients.
        chunks = iter(x.chunk(n_micro))
        handed = self._stage0(next(chunks))
        outs = []
        for chunk in chunks:
            outs.append(self._stage1(handed))
            handed = self._stage0(chunk)
        outs.append(self._stage1(handed))
        return torch.cat(outs)

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
    in_chans: int = 3,
    dropout: float = 0.1,
    img_size: int | None = None,
    micro_batches: int = 1,
) -> ModelParallelViT:
    """Build a ViT and wrap it for model parallelism across ``devices`` (pipelined in
    ``micro_batches`` micro-batches when > 1)."""
    if resolve_backbone(model_name) is not None:
        raise ValueError(f"model parallelism only splits timm ViT backbones, "
                         f"not '{model_name}' (registered or hf: backbone)")
    base = TimmClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        in_chans=in_chans,
        img_size=img_size,
    )
    return ModelParallelViT(base, devices=devices, split_block=split_block,
                            micro_batches=micro_batches)
