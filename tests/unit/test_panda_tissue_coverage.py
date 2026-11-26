"""Tissue coverage of PANDA mosaics (scripts/panda_tissue_coverage.py), pure functions."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "panda_tissue_coverage", Path(__file__).resolve().parents[2] / "scripts" / "panda_tissue_coverage.py")
cov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cov)


def test_tile_fractions_in_row_major_order():
    gray = np.full((4, 4), 255, np.uint8)       # 2x2 mosaic of 2 px tiles, all background
    gray[:2, :2] = 0                            # first tile all tissue
    gray[2:, 2:][0, 0] = 0                      # last tile, one pixel of four
    assert cov.tile_tissue_fractions(gray, 4).tolist() == [1.0, 0.0, 0.0, 0.25]
    with pytest.raises(ValueError):
        cov.tile_tissue_fractions(gray, 3)


def test_coverage_counts_tiles_and_kept_share():
    fr = np.array([1.0, 0.5, 0.5, 0.0])
    assert cov.coverage(fr, keep=1) == (3, pytest.approx(0.5))
    # mosaic order, not re-sorted: the smaller mosaic keeps the first tiles as they are
    assert cov.coverage(np.array([0.0, 1.0]), keep=1) == (1, pytest.approx(0.0))
    assert cov.coverage(fr, keep=4) == (3, pytest.approx(1.0))
    assert cov.coverage(np.zeros(4), keep=2) == (0, 1.0)     # no tissue: nothing is lost
