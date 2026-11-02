"""PANDA — Prostate cANcer graDe Assessment (Bulten et al., Nature Medicine 28, 2022).

10 616 prostate biopsies (whole-slide images from Radboud and Karolinska) graded
ISUP 0-5. The raw TIFFs (25-33 k px) are turned offline into one 1024 px mosaic per
biopsy by ``scripts/prepare_panda.py`` (see :mod:`src.data.wsi`), which also writes a
fixed ``splits.csv``; this module only reads those mosaics. Multiclass task with
ordered classes (``ordinal=True``: the ISUP grade), so the QWK metric is computed.

Config keys: ``data.root`` (folder with ``train.csv``; required) and optional
``data.mosaics`` (default ``<root>/mosaics_l1_16x256``), ``data.splits`` (default
``<root>/splits.csv``) and ``data.image_size`` (default 512: the mosaic is resized
inside the Dataset, since the transform factory only receives the split).

``train.csv`` carries no patient id, so the split is per biopsy (see
:func:`make_splits`). The reader exposes each sample's ``data_provider`` (the
centre) as ``PandaDataset.data_provider`` / :meth:`PandaDataset.groups`: the hook for
the Subgroups view (``groups.npz``, docs/inspeccion.md) once the monitor writes it.
Details and licence: docs/datasets/panda.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.registry import register_dataset
from src.data.spec import DatasetSpec

CLASSES = tuple(f"ISUP {g}" for g in range(6))
SPLITS = ("train", "val", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_SEED = 42
DEFAULT_MOSAIC_DIR = "mosaics_l1_16x256"
MOSAIC_EXT = ".jpg"
DEFAULT_IMAGE_SIZE = 512

_IMAGENET_MEAN, _IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ── Splits ────────────────────────────────────────────────────────────────────

def make_splits(labels: pd.DataFrame, fractions=SPLIT_FRACTIONS,
                seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Fixed train/val/test partition, stratified by (data_provider, isup_grade).

    ``labels`` is ``train.csv`` (``image_id``, ``data_provider``, ``isup_grade``).
    Each stratum is ordered by a hash of ``seed`` and the id, then cut at the given
    fractions. The order of one id never depends on the others, so the result does
    not depend on row order and is stable: dropping a few ids (unreadable TIFFs)
    only moves the biopsies next to a cut, instead of reshuffling the stratum.
    The partition is **per biopsy**: PANDA gives no patient id. Returns
    ``image_id, split`` sorted by id.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must add up to 1, got {fractions}")
    parts = []
    for _, group in sorted(labels.groupby(["data_provider", "isup_grade"]),
                           key=lambda kv: kv[0]):
        ids = np.array(sorted(group["image_id"], key=lambda i: _split_key(seed, i)))
        n = len(ids)
        n_train = round(n * fractions[0])
        n_val = round(n * fractions[1])
        split = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        parts.append(pd.DataFrame({"image_id": ids, "split": split}))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("image_id", ignore_index=True)


def _split_key(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()


# ── Dataset ───────────────────────────────────────────────────────────────────

class PandaDataset(Dataset):
    """One split of PANDA: the mosaic of each biopsy, resized to ``image_size``,
    with a one-hot ISUP target (6,)."""

    def __init__(self, root: str | Path, split: str = "train", transform=None,
                 image_size: int = DEFAULT_IMAGE_SIZE, mosaics: str | Path | None = None,
                 splits: str | Path | None = None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        root = Path(root).expanduser()
        self.mosaic_dir = Path(mosaics).expanduser() if mosaics else root / DEFAULT_MOSAIC_DIR
        meta = _read_meta(root, splits)
        meta = meta[meta["split"] == split].sort_values("image_id", ignore_index=True)
        missing = [i for i in meta["image_id"] if not self._path(i).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} mosaics of split {split!r} are missing in {self.mosaic_dir} "
                f"(e.g. {missing[0]}); run scripts/prepare_panda.py and rebuild the splits "
                f"(it leaves out the ids listed in errors.csv)")
        self.image_ids: list[str] = meta["image_id"].tolist()
        #: Centre of each sample ('karolinska' | 'radboud'), aligned with the samples.
        self.data_provider: list[str] = meta["data_provider"].tolist()
        grades = torch.as_tensor(meta["isup_grade"].to_numpy(), dtype=torch.long)
        self.targets = torch.nn.functional.one_hot(grades, len(CLASSES)).float()
        self.image_size = image_size
        self.transform = transform

    def _path(self, image_id: str) -> Path:
        return self.mosaic_dir / f"{image_id}{MOSAIC_EXT}"

    def groups(self) -> dict[str, np.ndarray]:
        """Grouping variables for ``groups.npz`` (Subgroups view), one entry per sample."""
        return {"data_provider": np.asarray(self.data_provider)}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self._path(self.image_ids[idx])) as im:
            im = im.convert("RGB")
            if im.size != (self.image_size, self.image_size):
                im = im.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            image = np.asarray(im)                  # (H, W, 3) uint8
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
        return image, self.targets[idx]


class RandomRot90:
    """Rotate a (C, H, W) tensor by a random multiple of 90 degrees."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return torch.rot90(img, int(torch.randint(0, 4, (1,))), dims=(-2, -1))


def get_transforms(split: str) -> transforms.Compose:
    """Train: flips and 90-degree rotations (histology has no orientation); every
    split: ImageNet normalization. The resize is done by the Dataset."""
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    if split == "train":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRot90(),
            normalize,
        ])
    return transforms.Compose([transforms.ToTensor(), normalize])


def _read_meta(root: Path, splits: str | Path | None) -> pd.DataFrame:
    """``train.csv`` joined with the split of each biopsy."""
    splits_path = Path(splits).expanduser() if splits else root / "splits.csv"
    return pd.read_csv(root / "train.csv").merge(pd.read_csv(splits_path), on="image_id")


def _build(data_cfg: dict, split: str, transform) -> PandaDataset:
    return PandaDataset(data_cfg["root"], split, transform,
                        image_size=int(data_cfg.get("image_size", DEFAULT_IMAGE_SIZE)),
                        mosaics=data_cfg.get("mosaics"), splits=data_cfg.get("splits"))


def class_counts(data_cfg: dict, split: str = "train") -> tuple[torch.Tensor, int]:
    """Biopsies per ISUP grade and total of a split (reads the CSVs, no image)."""
    meta = _read_meta(Path(data_cfg["root"]).expanduser(), data_cfg.get("splits"))
    grades = meta.loc[meta["split"] == split, "isup_grade"].to_numpy()
    counts = np.bincount(grades, minlength=len(CLASSES))
    return torch.as_tensor(counts, dtype=torch.float32), int(len(grades))


PANDA = register_dataset(DatasetSpec(
    name="panda",
    description="PANDA — prostate biopsies, ISUP grade 0-5 (mosaics of tissue tiles)",
    classes=CLASSES,
    task="multiclass",
    in_channels=3,
    required_keys=("root",),
    build=_build,
    transforms=get_transforms,
    class_counts=class_counts,
    ordinal=True,
))
