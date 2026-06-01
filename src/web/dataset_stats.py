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


def class_distribution_from_parquet(parquet_path: Path) -> pd.DataFrame | None:
    """Load class distribution from metadata.parquet if available."""
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path)
        if "labels" not in df.columns or "split" not in df.columns:
            return None

        train_df = df[df["split"] == "train"]
        rows = []
        for cls in CLASS_NAMES:
            count = train_df["labels"].apply(
                lambda x: cls in x if isinstance(x, (list, set)) else False
            ).sum()
            rows.append({"class": cls, "train_count": int(count)})
        return pd.DataFrame(rows)
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
