"""Tests de dataset_stats — distribución de clases (bug ndarray) e imágenes por clase."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.web.dataset_stats import (
    CLASS_NAMES,
    class_distribution_from_parquet,
    class_distribution_approximate,
    find_example_patches,
    load_rgb_image,
)


def _make_parquet_with_ndarray_labels(tmp: Path) -> Path:
    """Crea un parquet con labels como ndarray (como el BigEarthNet real)."""
    rows = [
        {"patch_id": "S2A_T1_00_01", "labels": np.array(["Arable land", "Pastures"]),
         "split": "train", "country": "Austria"},
        {"patch_id": "S2A_T1_00_02", "labels": np.array(["Arable land", "Marine waters"]),
         "split": "train", "country": "Spain"},
        {"patch_id": "S2A_T1_00_03", "labels": np.array(["Marine waters"]),
         "split": "train", "country": "Spain"},
        {"patch_id": "S2A_T1_00_04", "labels": np.array(["Pastures"]),
         "split": "validation", "country": "France"},
    ]
    df = pd.DataFrame(rows)
    path = tmp / "metadata.parquet"
    df.to_parquet(path)
    return path


# ── Bug del ndarray ───────────────────────────────────────────────────────────


def test_distribution_handles_ndarray_labels():
    """El bug original: labels como ndarray daba count=0 para todo."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_parquet_with_ndarray_labels(Path(tmp))
        df = class_distribution_from_parquet(path)

        assert df is not None, "No debe devolver None con labels ndarray válidas"
        assert df["train_count"].sum() > 0, "El conteo total NO debe ser 0"

        counts = dict(zip(df["class"], df["train_count"]))
        # Arable land aparece en 2 patches de train
        assert counts["Arable land"] == 2
        # Marine waters aparece en 2 patches de train (00_02, 00_03)
        assert counts["Marine waters"] == 2
        # Pastures aparece en 1 patch de train (00_01); el de validation no cuenta
        assert counts["Pastures"] == 1


def test_distribution_only_counts_train_split():
    """Solo el split train debe contarse."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_parquet_with_ndarray_labels(Path(tmp))
        df = class_distribution_from_parquet(path)
        counts = dict(zip(df["class"], df["train_count"]))
        # Pastures en validation (00_04) NO debe sumar
        assert counts["Pastures"] == 1


def test_distribution_returns_none_for_missing_file():
    df = class_distribution_from_parquet(Path("/nonexistent/metadata.parquet"))
    assert df is None


def test_distribution_returns_all_19_classes():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_parquet_with_ndarray_labels(Path(tmp))
        df = class_distribution_from_parquet(path)
        assert len(df) == len(CLASS_NAMES) == 19


def test_distribution_returns_none_when_all_zero():
    """Si ninguna etiqueta coincide con CLASS_NAMES → None (fallback)."""
    with tempfile.TemporaryDirectory() as tmp:
        df = pd.DataFrame([
            {"patch_id": "p1", "labels": np.array(["UnknownClass"]), "split": "train"},
        ])
        path = Path(tmp) / "meta.parquet"
        df.to_parquet(path)
        result = class_distribution_from_parquet(path)
        assert result is None


def test_approximate_distribution_is_fallback():
    df = class_distribution_approximate()
    assert len(df) == 19
    assert df["train_count"].sum() > 0


# ── find_example_patches ──────────────────────────────────────────────────────


def test_find_example_patches():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_parquet_with_ndarray_labels(Path(tmp))
        patches = find_example_patches(path, "Arable land", n=4)
        assert len(patches) == 2  # solo 2 patches de train tienen Arable land
        assert all("S2A" in p for p in patches)


def test_find_example_patches_missing_file():
    patches = find_example_patches(Path("/nonexistent.parquet"), "Arable land")
    assert patches == []


def test_find_example_patches_nonexistent_class():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_parquet_with_ndarray_labels(Path(tmp))
        patches = find_example_patches(path, "Beaches, dunes, sands", n=4)
        assert patches == []


# ── load_rgb_image ────────────────────────────────────────────────────────────


def test_load_rgb_image_missing_dir():
    img = load_rgb_image(Path("/nonexistent"), "S2A_T1_00_01")
    assert img is None


def test_load_rgb_image_with_synthetic_tifs():
    """Crea TIFs sintéticos y verifica que load_rgb_image los lee y estira."""
    rasterio = pytest.importorskip("rasterio")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        patch_id = "S2A_MSIL2A_20170613_R022_T33UUP_26_57"
        scene_id = "_".join(patch_id.rsplit("_", 2)[:-2])
        patch_dir = root / scene_id / patch_id
        patch_dir.mkdir(parents=True)

        for band in ("B04", "B03", "B02"):
            data = (np.random.rand(120, 120) * 3000).astype(np.float32)
            tif = patch_dir / f"{patch_id}_{band}.tif"
            with rasterio.open(
                tif, "w", driver="GTiff", height=120, width=120,
                count=1, dtype="float32",
            ) as dst:
                dst.write(data, 1)

        img = load_rgb_image(root, patch_id)
        assert img is not None
        assert img.shape == (120, 120, 3)
        assert img.dtype == np.uint8
        assert img.min() >= 0 and img.max() <= 255
