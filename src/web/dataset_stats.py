"""BigEarthNet-S2 dataset statistics for the Dataset Explorer tab.

Computes class distributions, co-occurrence, and split sizes from the
metadata.parquet file when available. Falls back to hardcoded constants
derived from the official BigEarthNet-S2 v2.0 release when the file is
not reachable (e.g. running the dashboard away from the cluster).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Official BigEarthNet-S2 v2.0 class labels ────────────────────────────────
CLASS_NAMES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# Approximate sample counts per class in BigEarthNet-S2 v2.0 (train split)
# Source: official BigEarthNet statistics
_APPROX_TRAIN_COUNTS = {
    "Urban fabric": 12_200,
    "Industrial or commercial units": 4_800,
    "Arable land": 57_000,
    "Permanent crops": 14_500,
    "Pastures": 23_500,
    "Complex cultivation patterns": 33_000,
    "Land principally occupied by agriculture": 3_800,
    "Agro-forestry areas": 11_200,
    "Broad-leaved forest": 40_000,
    "Coniferous forest": 48_000,
    "Mixed forest": 28_000,
    "Natural grassland and sparsely vegetated areas": 20_000,
    "Moors, heathland and sclerophyllous vegetation": 15_500,
    "Transitional woodland, shrub": 23_000,
    "Beaches, dunes, sands": 3_200,
    "Inland wetlands": 5_600,
    "Coastal wetlands": 2_100,
    "Inland waters": 13_500,
    "Marine waters": 8_400,
}

SPLIT_SIZES = {"train": 237_871, "val": 122_342, "test": 119_825}


# The metadata stores one class under its full CORINE name; CLASS_NAMES abbreviates
# it. Without this alias that class counts as 0 (name mismatch) in every parquet
# count — the per-class support and the Overview treemap.
_LABEL_ALIASES = {
    "Land principally occupied by agriculture, with significant areas of natural vegetation":
        "Land principally occupied by agriculture",
}


def _canon_label(lbl: str) -> str:
    return _LABEL_ALIASES.get(lbl, lbl)


def class_distribution_from_parquet(parquet_path: Path) -> pd.DataFrame | None:
    """Load class distribution from metadata.parquet if available.

    BigEarthNet metadata stores labels as numpy ndarrays (not lists), and the
    train split is named "train". We explode every label across all train
    patches and count occurrences vectorised for speed.
    """
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["labels", "split"])
        if "labels" not in df.columns or "split" not in df.columns:
            return None

        train_df = df[df["split"] == "train"]
        if train_df.empty:
            return None

        # labels is an ndarray per row → flatten all into one long list
        all_labels: list[str] = []
        for arr in train_df["labels"]:
            if arr is not None:
                all_labels.extend(_canon_label(x) for x in arr)

        counts = pd.Series(all_labels).value_counts()
        rows = [{"class": cls, "train_count": int(counts.get(cls, 0))}
                for cls in CLASS_NAMES]
        result = pd.DataFrame(rows)
        # If everything is zero something went wrong → signal fallback
        if result["train_count"].sum() == 0:
            return None
        return result
    except Exception:
        return None


def val_support_from_parquet(parquet_path: Path) -> dict[str, int] | None:
    """Per-class support in the VALIDATION split (how many val patches carry each
    class). Support is a property of the dataset, not of the model, so this lets
    the per-class view show it for runs already trained — no retraining needed.
    Returns {class_name: count} or None if the parquet/split is unavailable."""
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["labels", "split"])
        if "labels" not in df.columns or "split" not in df.columns:
            return None
        val_df = df[df["split"] == "validation"]
        if val_df.empty:
            return None
        all_labels: list[str] = []
        for arr in val_df["labels"]:
            if arr is not None:
                all_labels.extend(_canon_label(x) for x in arr)
        counts = pd.Series(all_labels).value_counts()
        return {cls: int(counts.get(cls, 0)) for cls in CLASS_NAMES}
    except Exception:
        return None


def class_distribution_approximate() -> pd.DataFrame:
    """Return approximate class distribution (hardcoded from official stats)."""
    return pd.DataFrame([
        {"class": cls, "train_count": _APPROX_TRAIN_COUNTS.get(cls, 0)}
        for cls in CLASS_NAMES
    ])


def cooccurrence_from_perclass(perclass_csv: Path) -> pd.DataFrame | None:
    """Approximate co-occurrence from per-class precision (proxy for confusion).

    Returns a CLASS × CLASS matrix where entry (i, j) estimates how often
    class i is confused with class j based on precision and recall patterns.
    This is not a true co-occurrence but is useful for understanding which
    classes the model mixes up.
    """
    if not perclass_csv.exists():
        return None
    try:
        df = pd.read_csv(perclass_csv)
        last_epoch = df["epoch"].max()
        ep_df = df[df["epoch"] == last_epoch].set_index("class_name")
        return ep_df[["f1", "precision", "recall"]].reindex(CLASS_NAMES)
    except Exception:
        return None


def get_country_distribution(parquet_path: Path) -> pd.Series | None:
    """Return patch count by country if metadata has that column."""
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["country", "split"])
        return df[df["split"] == "train"]["country"].value_counts()
    except Exception:
        return None


def find_example_patches(
    parquet_path: Path, class_name: str, n: int = 4, split: str = "train"
) -> list[str]:
    """Return up to n patch_ids whose labels contain class_name.

    Used by the dashboard to display example RGB images per class.
    """
    if not parquet_path.exists():
        return []
    try:
        df = pd.read_parquet(parquet_path, columns=["patch_id", "labels", "split"])
        df = df[df["split"] == split]
        matches = []
        for patch_id, arr in zip(df["patch_id"], df["labels"]):
            # Canonicalise labels so the abbreviated CLASS_NAMES match the full
            # CORINE names in the metadata (otherwise that class finds 0 patches
            # and the gallery shows 18 of 19).
            if arr is not None and class_name in {_canon_label(x) for x in arr}:
                matches.append(patch_id)
                if len(matches) >= n * 3:  # gather extra to allow random choice
                    break
        if len(matches) > n:
            import random
            matches = random.sample(matches, n)
        return matches[:n]
    except Exception:
        return []


def load_rgb_image(root: Path, patch_id: str) -> "np.ndarray | None":
    """Load a patch's RGB proxy (B04, B03, B02) as an (H, W, 3) uint8 array.

    Applies a percentile stretch for display (Sentinel-2 reflectance is dark
    by default). Returns None if the TIF files are not reachable.
    """
    try:
        import rasterio
    except ImportError:
        return None

    # patch dir: root/scene_id/patch_id/  where scene_id = patch_id sin _row_col
    scene_id = "_".join(patch_id.rsplit("_", 2)[:-2])
    patch_dir = Path(root) / scene_id / patch_id
    if not patch_dir.exists():
        return None

    bands = []
    for band in ("B04", "B03", "B02"):
        tif = patch_dir / f"{patch_id}_{band}.tif"
        if not tif.exists():
            return None
        try:
            with rasterio.open(tif) as src:
                bands.append(src.read(1).astype(np.float32))
        except Exception:
            return None

    img = np.stack(bands, axis=-1)
    # Percentile stretch per image for visibility (2nd–98th percentile)
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    return (img * 255).astype(np.uint8)
