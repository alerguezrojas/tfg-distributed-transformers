from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# BigEarthNet-S2 v2.0 land use classes (19 classes)
CLASSES = [
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

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# Sentinel-2 bands used as RGB proxy
RGB_BANDS = ["B04", "B03", "B02"]

# Map our split names to the parquet split names
SPLIT_MAP = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


class BigEarthNetDataset(Dataset):
    """PyTorch Dataset for BigEarthNet-S2 v2.0.

    Loads RGB patches (B04, B03, B02 bands) and multi-label targets.
    Labels and splits are read from the metadata.parquet file.
    Directory structure: root/scene_id/patch_id/*.tif
    """

    def __init__(
        self,
        root: str,
        metadata_path: str,
        split: str = "train",
        transform=None,
    ):
        """
        Args:
            root: Path to the BigEarthNet-S2 root directory.
            metadata_path: Path to the metadata.parquet file.
            split: One of 'train', 'val', 'test'.
            transform: Optional torchvision transforms to apply to each image.
        """
        self.root = Path(root)
        self.transform = transform

        parquet_split = SPLIT_MAP[split]
        df = pd.read_parquet(metadata_path)
        df = df[df["split"] == parquet_split].reset_index(drop=True)

        self.patch_ids = df["patch_id"].tolist()
        self.labels_list = df["labels"].tolist()

    def _get_patch_dir(self, patch_id: str) -> Path:
        """Resolve patch directory from patch_id (two-level structure).

        patch_id format: S2X_MSIL2A_<date>_<orbit>_<tile>_<row>_<col>
        scene_id is patch_id without the last two parts (_row_col).
        """
        scene_id = "_".join(patch_id.rsplit("_", 2)[:-2])
        return self.root / scene_id / patch_id

    def _load_image(self, patch_dir: Path) -> np.ndarray:
        """Load and stack RGB bands from TIF files into a (H, W, 3) array."""
        bands = []
        for band in RGB_BANDS:
            tif_path = patch_dir / f"{patch_dir.name}_{band}.tif"
            with rasterio.open(tif_path) as src:
                bands.append(src.read(1).astype(np.float32))
        image = np.stack(bands, axis=-1)
        image = image / 10000.0  # Sentinel-2 reflectance scale
        image = np.clip(image, 0, 1)
        return image

    def _load_labels(self, idx: int) -> torch.Tensor:
        """Convert label list to multi-hot tensor."""
        labels = torch.zeros(len(CLASSES), dtype=torch.float32)
        for label in self.labels_list[idx]:
            if label in CLASS_TO_IDX:
                labels[CLASS_TO_IDX[label]] = 1.0
        return labels

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch_id = self.patch_ids[idx]
        patch_dir = self._get_patch_dir(patch_id)

        image = self._load_image(patch_dir)
        labels = self._load_labels(idx)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)

        return image, labels


def get_transforms(split: str) -> transforms.Compose:
    """Return appropriate transforms for each split."""
    if split == "train":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
