"""Test-time augmentation by the symmetries of the square (dihedral group D4).

For images without a canonical orientation (histology, satellite tiles) the model's prediction
should not change when the image is flipped or rotated by 90°; averaging the probabilities of
those views is a cheap way to reduce its variance at evaluation time. Opt-in
(``scripts/eval.py --tta N``): for oriented images (chest X-rays, natural photos) it is wrong.
"""
from __future__ import annotations

import torch

#: Allowed number of views: identity; + horizontal flip; + vertical flip and 180° rotation;
#: + the four views that swap height and width (90° and 270° rotations, transposes).
TTA_VIEWS: tuple[int, ...] = (1, 2, 4, 8)


def dihedral_views(images: torch.Tensor, n: int) -> list[torch.Tensor]:
    """The first ``n`` views of a batch ``(N, C, H, W)`` under D4, identity first. With 8 the
    images must be square (the 90° rotations swap H and W)."""
    if n not in TTA_VIEWS:
        raise ValueError(f"tta must be one of {TTA_VIEWS}, got {n}")
    h, w = images.shape[-2:]
    if n == 8 and h != w:
        raise ValueError(f"tta=8 needs square images, got {h}x{w}")
    views = [images, images.flip(-1), images.flip(-2), images.flip(-1).flip(-2)]
    if n == 8:
        r = images.rot90(1, dims=(-2, -1))
        views += [r, r.flip(-1), r.flip(-2), r.flip(-1).flip(-2)]
    return views[:n]

