"""Automatic model profile — params and forward FLOPs of ANY timm model.

The analytic predictor (src/performance_model.py) only knows the models in its
curated MODEL_TABLE. For any other timm model this module measures what the
predictor needs from the model itself, without training and without a GPU:

* parameter count (millions);
* forward FLOPs per image, counted by ``torch.utils.flop_counter.FlopCounterMode``.

The model is instantiated on the ``meta`` device (shapes only, no weights are
allocated), so even vit_large profiles in a fraction of a second. If a model cannot
be built or run on ``meta`` it falls back to a real CPU instance.

Unit convention: FlopCounterMode counts a multiply-add as 2 FLOPs (vit_base =
35.1 G), but MODEL_TABLE — and timm's own tables — count it as 1 (vit_base = 17.6,
i.e. GMACs). The MFU of the predictor was calibrated against MODEL_TABLE, so the
profile reports the count halved, in the table's unit.

Attention is forced through the "math" SDPA kernel: it decomposes into matmuls the
counter sees (the fused CPU kernel is not counted, which under-counts ViTs by ~4%).

This module imports torch and timm; the predictor itself stays pure and receives
the result as a ModelSpec (``resolve_model_spec``).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.performance_model import MODEL_TABLE, ModelSpec, spec_from_profile

# FlopCounterMode counts a multiply-add as 2 FLOPs; MODEL_TABLE counts it as 1.
_FLOPS_PER_MAC: int = 2
_DEVICES: tuple[str, ...] = ("meta", "cpu")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    params_m: float          # millions of parameters (with this in_chans/num_classes head)
    gflops_fwd: float        # forward GFLOPs per image, multiply-add = 1 (MODEL_TABLE unit)
    n_conv_layers: int       # Conv1d/2d/3d modules (a ViT has 1: the patch embedding)
    img_size: int
    in_chans: int
    num_classes: int
    device: str              # where it was measured: "meta" or "cpu"

    @property
    def convolutional(self) -> bool:
        """Convolutions beyond the patch-embedding stem → a CNN-like backbone."""
        return self.n_conv_layers > 1


def _build(name: str, device: str, in_chans: int, num_classes: int, img_size: int | None):
    import timm
    import torch

    kwargs = {"pretrained": False, "in_chans": in_chans, "num_classes": num_classes}
    if img_size is not None:
        kwargs["img_size"] = img_size
    with torch.device(device):
        return timm.create_model(name, **kwargs)


def _count_forward_flops(model, device: str, in_chans: int, img_size: int) -> int:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.zeros(1, in_chans, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        model(x)
    return counter.get_total_flops()


def _measure(name: str, device: str, in_chans: int, num_classes: int,
             img_size: int) -> ModelProfile:
    import torch

    model = _build(name, device, in_chans, num_classes, None).eval()
    try:
        flops = _count_forward_flops(model, device, in_chans, img_size)
    except (AssertionError, RuntimeError, ValueError):
        # Fixed-resolution models (ViT/Swin at 384…) reject other input sizes:
        # rebuild them for the requested resolution.
        model = _build(name, device, in_chans, num_classes, img_size).eval()
        flops = _count_forward_flops(model, device, in_chans, img_size)
    params = sum(p.numel() for p in model.parameters())
    n_conv = sum(isinstance(m, torch.nn.modules.conv._ConvNd) for m in model.modules())
    return ModelProfile(name=name, params_m=params / 1e6,
                        gflops_fwd=flops / _FLOPS_PER_MAC / 1e9, n_conv_layers=n_conv,
                        img_size=img_size, in_chans=in_chans, num_classes=num_classes,
                        device=device)


@lru_cache(maxsize=64)
def profile_model(name: str, in_chans: int = 3, num_classes: int = 1000,
                  img_size: int = 224) -> ModelProfile:
    """Params and forward FLOPs (multiply-add = 1) of a timm model at ``img_size``².

    Tries the meta device first and falls back to CPU. Raises ``ValueError`` if timm
    does not know the model or it cannot be run on either device."""
    import timm

    if not timm.is_model(name):
        raise ValueError(f"unknown timm model: {name}")
    errors = []
    for device in _DEVICES:
        try:
            return _measure(name, device, in_chans, num_classes, img_size)
        except Exception as exc:  # any build/forward failure → try the next device
            errors.append(f"{device}: {type(exc).__name__}: {exc}")
    raise ValueError(f"could not profile {name}: " + " | ".join(errors))


_TABLE_IN_CHANS: int = 3     # the curated entries are 3-channel, 224² models
_TABLE_IMG_SIZE: int = 224


def _base_name(name: str) -> str:
    """Architecture name without the pretrained-weights tag ("vit_base_patch16_224.augreg_in1k")."""
    return name.strip().split(".", 1)[0]


def _table_entry(name: str) -> ModelSpec | None:
    """EXACT MODEL_TABLE lookup (a trailing pretrained tag "name.tag" is dropped).

    Deliberately not ``model_spec()``: that one matches by substring (right for GPU
    names), so "wide_resnet50_2" would silently get resnet50's entry (26M params
    instead of ~69M) and be reported as calibrated."""
    return MODEL_TABLE.get(_base_name(name))


def resolve_model_spec(name: str, in_chans: int = 3, num_classes: int = 19,
                       img_size: int = 224) -> ModelSpec | None:
    """ModelSpec for the predictor: the curated MODEL_TABLE entry when ``name`` IS in
    the table at its 3-channel 224² setup (unchanged behaviour), otherwise a spec built
    from an automatic profile (``calibrated=False``). A table model asked for a
    different ``in_chans`` or ``img_size`` is profiled with them (its table numbers no
    longer apply). Returns None if the model cannot be profiled.

    ``num_classes`` defaults to the 19 BigEarthNet classes the predictor's dataset
    sizes assume; the head is a negligible part of params/FLOPs, so it alone never
    invalidates a table entry."""
    spec = _table_entry(name)
    if spec is not None and in_chans == _TABLE_IN_CHANS and img_size == _TABLE_IMG_SIZE:
        return spec
    try:
        prof = profile_model(_base_name(name), in_chans=in_chans, num_classes=num_classes,
                             img_size=img_size)
    except (ValueError, ImportError):
        return None
    return spec_from_profile(prof.name, round(prof.params_m, 2), round(prof.gflops_fwd, 3),
                             convolutional=prof.convolutional)
