import json
from pathlib import Path

import numpy as np
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


class BigEarthNetDataset(Dataset):
    """PyTorch Dataset for BigEarthNet-S2 v2.0.

    Loads RGB patches (B04, B03, B02 bands) and multi-label targets
    from the BigEarthNet-S2 directory structure.
    """

    # Sentinel-2 bands used as RGB proxy
    RGB_BANDS = ["B04", "B03", "B02"]

    def __init__(self, root: str, split: str = "train", transform=None):
        """
        Args:
            root: Path to the BigEarthNet-S2 root directory.
            split: One of 'train', 'val', 'test'.
            transform: Optional torchvision transforms to apply to each image.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.patches = self._load_split()

    def _load_split(self) -> list[str]:
        """Load patch names for the requested split from the metadata CSV."""
        split_file = self.root / "metadata" / f"{self.split}.csv"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                "Make sure BigEarthNet-S2 is fully extracted."
            )
        with open(split_file) as f:
            # First line is header
            patches = [line.strip() for line in f if line.strip() and not line.startswith("patch")]
        return patches

    def _load_labels(self, patch_dir: Path) -> torch.Tensor:
        """Read labels from the patch JSON metadata file."""
        label_file = patch_dir / "labels_metadata.json"
        with open(label_file) as f:
            meta = json.load(f)
        labels = torch.zeros(len(CLASSES), dtype=torch.float32)
        for label in meta["labels"]:
            if label in CLASS_TO_IDX:
                labels[CLASS_TO_IDX[label]] = 1.0
        return labels

    def _load_image(self, patch_dir: Path) -> np.ndarray:
        """Load and stack RGB bands from TIF files into a (H, W, 3) array."""
        bands = []
        for band in self.RGB_BANDS:
            tif_path = patch_dir / f"{patch_dir.name}_{band}.tif"
            with rasterio.open(tif_path) as src:
                bands.append(src.read(1).astype(np.float32))

        # Stack bands → (H, W, 3) then normalize to [0, 1]
        image = np.stack(bands, axis=-1)
        image = image / 10000.0  # Sentinel-2 reflectance scale
        image = np.clip(image, 0, 1)
        return image

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch_name = self.patches[idx]
        patch_dir = self.root / patch_name

        image = self._load_image(patch_dir)
        labels = self._load_labels(patch_dir)

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1)

        return image, labels


def get_transforms(split: str) -> transforms.Compose:
    """Return appropriate transforms for each split."""
    if split == "train":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
