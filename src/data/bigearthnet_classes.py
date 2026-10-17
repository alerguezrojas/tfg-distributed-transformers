"""BigEarthNet-S2 v2.0 constants — pure Python (no torch/rasterio), so that
light consumers such as the web dashboard can import them cheaply.
"""

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
