"""Data augmentation utilities for training.

Applied at the batch level in train_epoch, not in the DataLoader transform.
"""

from __future__ import annotations

import torch
from torch import Tensor


def mixup_batch(images: Tensor, labels: Tensor, alpha: float = 0.2) -> tuple[Tensor, Tensor]:
    """Alpha-blend two random pairings within a batch (mixup augmentation).

    Compatible with multi-label classification: labels are soft-blended too.
    Loss function (BCEWithLogitsLoss) natively handles soft labels in [0, 1].

    Args:
        images: (N, C, H, W) batch of images
        labels: (N, num_classes) float labels in {0, 1}
        alpha:  Beta distribution parameter. Higher α → more mixing.

    Returns:
        (mixed_images, mixed_labels) with the same shapes as inputs.
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[idx]
    mixed_labels = lam * labels + (1.0 - lam) * labels[idx]
    return mixed_images, mixed_labels
