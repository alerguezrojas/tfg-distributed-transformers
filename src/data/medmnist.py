"""MedMNIST — standardized biomedical image classification datasets.

Yang et al., "MedMNIST v2", Scientific Data 10:41 (2023); https://medmnist.com.
The official ``.npz`` files (Zenodo record 10519652) are read directly with NumPy,
so the ``medmnist`` package is not needed. Each file holds
``{train,val,test}_{images,labels}``; images are uint8, (N, H, W) for grayscale
or (N, H, W, 3) for RGB, at 28 px (``<flag>.npz``) or 64/128/224 px
(``<flag>_<size>.npz``).

Registered subsets (one dataset each, ``data.name``):

=====================  ==========  =====  =======  ===========================
name                   task        ch.    classes  content
=====================  ==========  =====  =======  ===========================
``medmnist_breast``    binary      1      1        breast ultrasound, malignant
``medmnist_derma``     multiclass  3      7        dermatoscopy (incl. melanoma)
``medmnist_path``      multiclass  3      9        colorectal cancer histology
``medmnist_chest``     multilabel  1      14       chest X-ray (NIH ChestX-ray14)
=====================  ==========  =====  =======  ===========================

Config keys: ``data.root`` (folder with the ``.npz`` files) and optional
``data.size`` (28 by default). No augmentation is applied, as in the MedMNIST
benchmark protocol, so results are comparable with the published baselines.
Details and licences: docs/datasets/medmnist.md.
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.registry import register_dataset
from src.data.spec import DatasetSpec

ZENODO_FILES = "https://zenodo.org/records/10519652/files"
SIZES = (28, 64, 128, 224)
DEFAULT_SIZE = 28

_IMAGENET_MEAN, _IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class Subset:
    """One MedMNIST subset as published (labels, MD5 per image size, licence)."""

    flag: str
    description: str
    task: str
    channels: int
    labels: tuple[str, ...]          # original MedMNIST class names, by label id
    md5: dict[int, str]              # image size → MD5 of its .npz
    license: str
    positive_label: int | None = None  # binary: original label id of the positive class

    @property
    def classes(self) -> tuple[str, ...]:
        """Model outputs: one column per class (just the positive one if binary)."""
        if self.task == "binary":
            return (self.labels[self.positive_label],)
        return self.labels


# Source: medmnist 3.0.2 (medmnist/info.py).
SUBSETS: dict[str, Subset] = {
    "medmnist_breast": Subset(
        flag="breastmnist", description="BreastMNIST — breast ultrasound, malignant vs. not",
        task="binary", channels=1, labels=("malignant", "normal, benign"), positive_label=0,
        md5={28: "750601b1f35ba3300ea97c75c52ff8f6", 64: "742edef2a1fd1524b2efff4bd7ba9364",
             128: "363e4b3f8d712e9b5de15470a2aaadf1", 224: "b56378a6eefa9fed602bb16d192d4c8b"},
        license="CC BY 4.0",
    ),
    "medmnist_derma": Subset(
        flag="dermamnist", description="DermaMNIST — dermatoscopic skin lesions (HAM10000)",
        task="multiclass", channels=3,
        labels=("actinic keratoses and intraepithelial carcinoma", "basal cell carcinoma",
                "benign keratosis-like lesions", "dermatofibroma", "melanoma",
                "melanocytic nevi", "vascular lesions"),
        md5={28: "0744692d530f8e62ec473284d019b0c7", 64: "b70a2f5635c6199aeaa28c31d7202e1f",
             128: "2defd784463fa5243564e855ed717de1", 224: "8974907d8e169bef5f5b96bc506ae45d"},
        license="CC BY-NC 4.0",
    ),
    "medmnist_path": Subset(
        flag="pathmnist", description="PathMNIST — colorectal cancer histology",
        task="multiclass", channels=3,
        labels=("adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
                "normal colon mucosa", "cancer-associated stroma",
                "colorectal adenocarcinoma epithelium"),
        md5={28: "a8b06965200029087d5bd730944a56c1", 64: "55aa9c1e0525abe5a6b9d8343a507616",
             128: "ac42d08fb904d92c244187169d1fd1d9", 224: "2c51a510bcdc9cf8ddb2af93af1eadec"},
        license="CC BY 4.0",
    ),
    "medmnist_chest": Subset(
        flag="chestmnist", description="ChestMNIST — chest X-ray, 14 findings (ChestX-ray14)",
        task="multilabel", channels=1,
        labels=("atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule",
                "pneumonia", "pneumothorax", "consolidation", "edema", "emphysema",
                "fibrosis", "pleural", "hernia"),
        md5={28: "02c8a6516a18b556561a56cbdd36c4a8", 64: "9de6cd0b934ebb5b7426cfba5efbae16",
             128: "db107e5590b27930b62dbaf558aebee3", 224: "45bd33e6f06c3e8cdb481c74a89152aa"},
        license="CC BY 4.0",
    ),
}


# ── Files ─────────────────────────────────────────────────────────────────────

def npz_filename(subset: Subset, size: int = DEFAULT_SIZE) -> str:
    if size not in SIZES:
        raise ValueError(f"MedMNIST size must be one of {SIZES}, got {size!r}")
    return f"{subset.flag}.npz" if size == DEFAULT_SIZE else f"{subset.flag}_{size}.npz"


def download_url(subset: Subset, size: int = DEFAULT_SIZE) -> str:
    return f"{ZENODO_FILES}/{npz_filename(subset, size)}?download=1"


def _md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)      # integrity check of a published file
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(subset: Subset, root: str | Path, size: int = DEFAULT_SIZE) -> Path:
    """Download the subset's .npz into ``root`` (skipped if already valid) and
    check its MD5 against the published one."""
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / npz_filename(subset, size)
    expected = subset.md5[size]
    if path.exists() and _md5(path) == expected:
        return path
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(download_url(subset, size), tmp)
    got = _md5(tmp)
    if got != expected:
        tmp.unlink(missing_ok=True)
        raise IOError(f"MD5 mismatch for {path.name}: got {got}, expected {expected}")
    tmp.replace(path)
    return path


# ── Dataset ───────────────────────────────────────────────────────────────────

def encode_targets(raw: np.ndarray, subset: Subset) -> torch.Tensor:
    """MedMNIST labels → float targets (N, C) in the task's format."""
    raw = np.asarray(raw)
    if subset.task == "multilabel":                 # already multi-hot (N, C)
        return torch.as_tensor(raw, dtype=torch.float32)
    ids = torch.as_tensor(raw.reshape(-1), dtype=torch.long)
    if subset.task == "binary":                     # (N, 1): is it the positive class?
        return (ids == subset.positive_label).float().unsqueeze(1)
    return torch.nn.functional.one_hot(ids, len(subset.labels)).float()


class MedMNISTDataset(Dataset):
    """One split of a MedMNIST subset, held in memory (uint8 images)."""

    _SPLIT_KEYS = {"train": "train", "val": "val", "test": "test"}

    def __init__(self, npz_path: str | Path, subset: Subset, split: str = "train",
                 transform=None):
        key = self._SPLIT_KEYS[split]
        with np.load(npz_path) as data:
            self.images = data[f"{key}_images"]
            self.targets = encode_targets(data[f"{key}_labels"], subset)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]                    # (H, W) or (H, W, 3), uint8
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.ascontiguousarray(image))
        return image, self.targets[idx]


def make_transforms(channels: int, image_size: int = 224):
    """Same transform for every split (no augmentation, as the MedMNIST benchmark):
    to [0, 1], resize to the model's input size and normalize — ImageNet statistics
    for RGB (pretrained backbones), mean/std 0.5 for grayscale."""
    mean, std = (_IMAGENET_MEAN, _IMAGENET_STD) if channels == 3 else ([0.5], [0.5])
    tfm = transforms.Compose([
        transforms.ToTensor(),                      # HWC/HW uint8 → CHW float in [0, 1]
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Normalize(mean=mean, std=std),
    ])
    return lambda split: tfm


def _npz_path(subset: Subset, data_cfg: dict) -> Path:
    size = int(data_cfg.get("size", DEFAULT_SIZE))
    return Path(data_cfg["root"]).expanduser() / npz_filename(subset, size)


def _spec(name: str, subset: Subset) -> DatasetSpec:
    def build(data_cfg: dict, split: str, transform) -> MedMNISTDataset:
        return MedMNISTDataset(_npz_path(subset, data_cfg), subset, split, transform)

    def class_counts(data_cfg: dict, split: str = "train") -> tuple[torch.Tensor, int]:
        with np.load(_npz_path(subset, data_cfg)) as data:
            targets = encode_targets(data[f"{split}_labels"], subset)
        return targets.sum(dim=0), len(targets)

    return DatasetSpec(
        name=name,
        description=subset.description,
        classes=subset.classes,
        task=subset.task,
        in_channels=subset.channels,
        required_keys=("root",),
        build=build,
        transforms=make_transforms(subset.channels),
        class_counts=class_counts,
    )


SPECS: dict[str, DatasetSpec] = {
    name: register_dataset(_spec(name, subset)) for name, subset in SUBSETS.items()
}
