"""Unit tests for src/estimation_models.py — selectable speedup laws."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.estimation_models import (
    SPEEDUP_MODELS, amdahl, gustafson, linear, speedup_curve,
)


def test_linear_is_identity():
    assert linear(1) == 1
    assert linear(4) == 4
    assert linear(8) == 8


def test_amdahl_bounds():
    # At n=1 every model returns 1x.
    assert amdahl(1, 0.05) == pytest.approx(1.0)
    # Amdahl is always below linear for n>1.
    assert amdahl(4, 0.05) < 4
    # It plateaus at 1/s as n -> infinity.
    assert amdahl(10_000, 0.10) == pytest.approx(10.0, rel=0.02)


def test_amdahl_zero_serial_equals_linear():
    for n in (1, 2, 4, 8):
        assert amdahl(n, 0.0) == pytest.approx(linear(n))


def test_gustafson_near_linear():
    assert gustafson(1, 0.05) == pytest.approx(1.0)
    # With a small serial fraction it stays close to (but below) linear.
    assert gustafson(8, 0.05) == pytest.approx(8 - 0.05 * 7)
    assert gustafson(8, 0.05) < 8


def test_ordering_amdahl_le_gustafson_le_linear():
    for n in (2, 4, 8):
        s = 0.1
        assert amdahl(n, s) <= gustafson(n, s) <= linear(n) + 1e-9


def test_registry_has_three_models_with_formulas():
    assert set(SPEEDUP_MODELS) == {"linear", "amdahl", "gustafson"}
    for m in SPEEDUP_MODELS.values():
        assert m.name and m.formula and m.description
        assert callable(m.fn)


def test_speedup_curve_matches_model():
    curve = speedup_curve("amdahl", [1, 2, 4, 8], serial_fraction=0.05)
    assert curve == [pytest.approx(amdahl(n, 0.05)) for n in (1, 2, 4, 8)]


def test_serial_fraction_clamped():
    # Out-of-range serial fractions must not crash or produce nonsense.
    assert amdahl(4, -1.0) == pytest.approx(linear(4))   # clamped to 0
    assert amdahl(4, 2.0) == pytest.approx(1.0)           # clamped to 1 -> no speedup
