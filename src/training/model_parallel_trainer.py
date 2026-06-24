"""ModelParallelTrainer — single-process trainer for a model split across devices.

Pipeline (naive) model parallelism places the first stages of the network on one
device and the rest on another (see :class:`src.models.model_parallel.ModelParallelViT`).
The model is therefore *already* placed across its stage devices, so — unlike the
plain :class:`Trainer` — we must NOT move it as a whole (that would collapse the
split onto a single device).

Inputs and labels live on the model's OUTPUT device (the forward routes the input
to the first stage internally and returns the logits on the last device), so the
rest is the standard single-process Trainer: the very same Template-Method loop
(``EpochController``) and the very same decorator stack (metric reporters,
per-class, confusion, batch monitor, layer hooks, energy) apply unchanged. This is
what makes model parallelism a first-class strategy alongside single/DDP/heterogeneous
rather than a bespoke training loop.
"""
from __future__ import annotations

import torch.nn as nn

from src.training.trainer import Trainer


class ModelParallelTrainer(Trainer):
    """Trainer for a :class:`ModelParallelViT` (model split across devices)."""

    def _place_model(self, model: nn.Module, device) -> nn.Module:
        # The model is already split across its stage devices by ModelParallelViT;
        # moving it as a whole would undo the split. Leave it exactly where it is.
        return model
