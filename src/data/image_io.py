"""Reading loose image files the way the datasets read theirs (used by ``paravit predict``).

``open_image`` converts to 8-bit grayscale or RGB like the datasets' own readers. Pillow's
``convert("L")`` CLIPS a 16-bit or float image at 255 (a 16-bit radiograph becomes almost
white), so high-bit-depth images are first rescaled to 0-255 with their own min-max range.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_HIGH_BIT_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")


def _to_8bit(im: Image.Image) -> Image.Image:
    arr = np.asarray(im, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    scaled = np.zeros_like(arr) if hi <= lo else (arr - lo) / (hi - lo) * 255.0
    return Image.fromarray(np.round(scaled).astype(np.uint8))           # 2-D uint8 → mode "L"


def open_image(path: str | Path, mode: str, size: int | None = None) -> np.ndarray:
    """``path`` as a uint8 array in ``mode`` ("L" → (H, W), "RGB" → (H, W, 3)), resized
    (bilinear) to ``size`` × ``size`` when given and different."""
    with Image.open(path) as im:
        if im.mode in _HIGH_BIT_MODES:
            im = _to_8bit(im)
        im = im.convert(mode)
        if size is not None and im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BILINEAR)
        return np.array(im)                         # a copy: asarray would be read-only
