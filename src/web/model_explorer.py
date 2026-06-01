"""Model Explorer — timm model browser with parameter and VRAM estimates.

Provides lightweight model info (parameters, FLOPs, estimated VRAM) without
loading full weights. Uses torchinfo for a single forward pass on CPU with a
tiny batch to measure actual FLOPs and activation memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import timm


# Models curated for this project (representative subset)
CURATED_MODELS = {
    "ViT": [
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224",
    ],
    "Swin": [
        "swin_tiny_patch4_window7_224",
        "swin_small_patch4_window7_224",
        "swin_base_patch4_window7_224",
    ],
    "ResNet": [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "wide_resnet50_2",
    ],
    "EfficientNet": [
        "efficientnet_b0",
        "efficientnet_b2",
        "efficientnet_b4",
        "efficientnetv2_s",
        "efficientnetv2_m",
    ],
    "ConvNeXt": [
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
    ],
    "MobileNet": [
        "mobilenetv3_small_100",
        "mobilenetv3_large_100",
        "mobilenetv2_100",
    ],
}

ALL_FAMILIES = list(CURATED_MODELS.keys())


@dataclass
class ModelStats:
    name: str
    family: str
    total_params_m: float
    trainable_params_m: float
    flops_mflops: float | None
    weight_mb: float
    gradient_mb: float
    optimizer_mb: float    # AdamW: 2× weights
    activation_mb_per_img: float | None
    total_static_gb: float

    def vram_estimate_gb(self, batch_size: int) -> float:
        act = (self.activation_mb_per_img or 40.0) * batch_size
        return (self.weight_mb + self.gradient_mb + self.optimizer_mb + act) / 1024

    @property
    def total_static_mb(self) -> float:
        return self.weight_mb + self.gradient_mb + self.optimizer_mb


@lru_cache(maxsize=32)
def get_model_stats(model_name: str, num_classes: int = 19) -> ModelStats | None:
    """Compute model stats for a given timm model name.

    Results are cached so repeated calls are free.
    """
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        weight_mb = total_params * 4 / 1e6       # float32
        gradient_mb = weight_mb                   # one gradient per param
        optimizer_mb = weight_mb * 2              # AdamW: m + v

        # Quick FLOPs + activation estimate via torchinfo
        flops_mflops = None
        activation_mb_per_img = None
        try:
            from torchinfo import summary
            dummy = torch.zeros(1, 3, 224, 224)
            info = summary(model, input_data=dummy, verbose=0)
            flops_mflops = info.total_mult_adds / 1e6
            # Activation memory ≈ total output bytes of all layers
            activation_mb_per_img = info.total_output_bytes / 1e6
        except Exception:
            pass

        family = _detect_family(model_name)

        return ModelStats(
            name=model_name,
            family=family,
            total_params_m=total_params / 1e6,
            trainable_params_m=trainable_params / 1e6,
            flops_mflops=flops_mflops,
            weight_mb=weight_mb,
            gradient_mb=gradient_mb,
            optimizer_mb=optimizer_mb,
            activation_mb_per_img=activation_mb_per_img,
            total_static_gb=(weight_mb + gradient_mb + optimizer_mb) / 1024,
        )
    except Exception:
        return None


def _detect_family(name: str) -> str:
    name_l = name.lower()
    for family, models in CURATED_MODELS.items():
        if any(m in name_l for m in [family.lower(), name_l]):
            if name in models:
                return family
    if "vit" in name_l or "deit" in name_l:
        return "ViT"
    if "swin" in name_l:
        return "Swin"
    if "resnet" in name_l or "wide_resnet" in name_l:
        return "ResNet"
    if "efficientnet" in name_l:
        return "EfficientNet"
    if "convnext" in name_l:
        return "ConvNeXt"
    if "mobile" in name_l:
        return "MobileNet"
    return "Other"


def compare_models(
    model_names: list[str],
    batch_sizes: list[int],
    num_classes: int = 19,
) -> list[dict]:
    """Return a list of dicts suitable for a comparison DataFrame."""
    rows = []
    for name in model_names:
        stats = get_model_stats(name, num_classes)
        if stats is None:
            continue
        row: dict = {
            "Model": name,
            "Family": stats.family,
            "Params (M)": round(stats.total_params_m, 1),
            "FLOPs (MFLOPs)": round(stats.flops_mflops, 0) if stats.flops_mflops else "—",
            "Weights (MB)": round(stats.weight_mb, 0),
            "Static (GB)": round(stats.total_static_gb, 2),
        }
        for bs in batch_sizes:
            row[f"VRAM est. bs={bs} (GB)"] = round(stats.vram_estimate_gb(bs), 2)
        rows.append(row)
    return rows
