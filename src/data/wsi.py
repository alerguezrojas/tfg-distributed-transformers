"""Whole-slide-image helpers: tissue-aware mosaics from a pyramidal TIFF.

A biopsy of PANDA is a 25-33 k px pyramidal TIFF: far too big for a ViT. The
offline recipe is to take a reduced pyramid level, cut it into square tiles, keep
the ``n_tiles`` with the most tissue and paste them in a ``k x k`` mosaic that the
model sees as a single image. Everything here is pure NumPy (no torch, no GPU);
only :func:`read_level` touches the disk, through ``tifffile``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

WHITE = 255     # glass background; padding and missing tiles use it


def read_level(path: str | Path, level: int) -> np.ndarray:
    """Pyramid ``level`` of a TIFF as an (H, W, 3) uint8 array (0 = full resolution)."""
    import tifffile  # imported here so the pure functions work without the extra deps
    with tifffile.TiffFile(path) as tif:
        levels = tif.series[0].levels
        if level >= len(levels):
            raise ValueError(f"{Path(path).name}: level {level} requested but the "
                             f"file has {len(levels)}")
        return levels[level].asarray()


def grid_side(n_tiles: int) -> int:
    """Side of the square mosaic grid; ``n_tiles`` must be a perfect square."""
    side = math.isqrt(n_tiles)
    if side < 1 or side * side != n_tiles:
        raise ValueError(f"n_tiles must be a perfect square, got {n_tiles}")
    return side


def pad_to_multiple(img: np.ndarray, tile: int) -> np.ndarray:
    """Pad bottom/right with white so both sides are multiples of ``tile``."""
    h, w = img.shape[:2]
    ph, pw = (-h) % tile, (-w) % tile
    if ph == 0 and pw == 0:
        return img
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), constant_values=WHITE)


def tissue_scores(img: np.ndarray, tile: int) -> np.ndarray:
    """Darkness of every tile (row-major, flat): the all-white pixel sum minus the
    tile's own, so more tissue (less white) means a higher score.
    ``img`` must already be a multiple of ``tile`` on both sides."""
    h, w, c = img.shape
    rows, cols = h // tile, w // tile
    sums = img.reshape(rows, tile, cols, tile, c).sum(axis=(1, 3, 4), dtype=np.int64)
    return (WHITE * tile * tile * c - sums).reshape(-1)


def select_tiles(scores: np.ndarray, n_tiles: int) -> np.ndarray:
    """Indices of the ``n_tiles`` highest-scoring tiles, best first (ties by position,
    so the choice is deterministic). Fewer if the image has fewer tiles."""
    return np.argsort(-scores, kind="stable")[:n_tiles]


def extract_tiles(img: np.ndarray, tile: int, indices: np.ndarray) -> np.ndarray:
    """The tiles ``indices`` (row-major over the grid) as (len(indices), tile, tile, C)."""
    h, w, c = img.shape
    cols = w // tile
    out = np.empty((len(indices), tile, tile, c), dtype=img.dtype)
    for k, idx in enumerate(indices):
        r, q = divmod(int(idx), cols)
        out[k] = img[r * tile:(r + 1) * tile, q * tile:(q + 1) * tile]
    return out


def assemble_mosaic(tiles: np.ndarray, n_tiles: int) -> np.ndarray:
    """Paste ``tiles`` row by row into a ``k x k`` mosaic; if there are fewer than
    ``n_tiles`` the rest are white tiles."""
    side = grid_side(n_tiles)
    n, tile, _, c = tiles.shape
    if n > n_tiles:
        raise ValueError(f"got {n} tiles for a mosaic of {n_tiles}")
    if n < n_tiles:
        blank = np.full((n_tiles - n, tile, tile, c), WHITE, dtype=tiles.dtype)
        tiles = np.concatenate([tiles, blank])
    return (tiles.reshape(side, side, tile, tile, c)
                 .transpose(0, 2, 1, 3, 4)
                 .reshape(side * tile, side * tile, c))


def make_mosaic(img: np.ndarray, tile: int = 256, n_tiles: int = 16) -> np.ndarray:
    """(H, W, 3) image → (k*tile, k*tile, 3) mosaic of its ``n_tiles`` most
    tissue-rich tiles."""
    grid_side(n_tiles)                       # fail early on a bad n_tiles
    img = pad_to_multiple(img, tile)
    idx = select_tiles(tissue_scores(img, tile), n_tiles)
    return assemble_mosaic(extract_tiles(img, tile, idx), n_tiles)


def tissue_fraction(mosaic: np.ndarray, threshold: int = 230) -> float:
    """Share of pixels whose mean channel value is below ``threshold`` (not glass)."""
    return float((mosaic.mean(axis=-1) < threshold).mean())
