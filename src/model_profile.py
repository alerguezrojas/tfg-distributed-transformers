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
    try:        # feature of the three-term throughput model; optional (falls back to MFU)
        saved = round(_saved_gb_at(_base_name(name), in_chans, num_classes, img_size), 4)
    except (ValueError, ImportError):
        saved = None
    return spec_from_profile(prof.name, round(prof.params_m, 2), round(prof.gflops_fwd, 3),
                             convolutional=prof.convolutional, saved_gb_per_img=saved)


# Above this resolution the saved activations are measured at it and scaled by the pixel
# count: profiling is a real CPU forward that keeps every activation (vit_large at 1024 px
# would hold ~6.5 GB inside the web process). The scaling is near exact for the architectures
# measured here (224 → 512 px: ×5.2 against (512/224)² = 5.22 for every MODEL_TABLE model),
# since flash attention keeps no N×N matrix and convolution maps grow with the pixels.
SAVED_MEASURE_MAX_PX: int = 512


def _saved_gb_at(name: str, in_chans: int, num_classes: int, img_size: int) -> float:
    """``saved_activation_gb`` at ``img_size``, measured at ``SAVED_MEASURE_MAX_PX`` and scaled
    by the pixel ratio when the requested size is larger."""
    if img_size <= SAVED_MEASURE_MAX_PX:
        return saved_activation_gb(name, in_chans, num_classes, img_size)
    base = saved_activation_gb(name, in_chans, num_classes, SAVED_MEASURE_MAX_PX)
    return base * (img_size / SAVED_MEASURE_MAX_PX) ** 2


# ── Activation memory saved for the backward pass ────────────────────────────────
# FLOPs alone do not set the training time of every architecture: hierarchical
# transformers (Swin), CNNs and above all depthwise CNNs (EfficientNet) move far more
# memory per FLOP than a ViT. The bytes autograd keeps for the backward pass are a
# direct, GPU-free measure of that traffic. Measured on CPU (meta tensors have no
# storage to count) with one image, fp32. Under AMP part of it is stored in half
# precision (LayerNorm and softmax stay fp32), so "about half" is an ASSUMPTION, not a
# measurement. Attention runs through the flash SDPA kernel, which does not keep the
# N×N attention matrix, like the fused kernels used on the GPU (the "math" kernel
# would keep it and double a ViT's figure at 512 px).
# See docs/performance_model.md, "Arquitecturas distintas de ViT".


def _count_saved_bytes(model, in_chans: int, img_size: int) -> int:
    import torch

    # Parameters are excluded: they are read once per BATCH, not once per image.
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen: set[tuple[int, int]] = set()
    # The saved tensors stay alive here until the forward ends (so no address is reused and
    # the dedup by address holds), but the graph keeps only None: returning ``t`` itself
    # from the pack hook makes a tensor ↔ grad_fn cycle that autograd never frees, and
    # every model profiled leaked all its activations (~0.5-1 GB each at 512 px).
    keep: list = []
    total = 0

    def pack(t):
        nonlocal total
        storage = t.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in seen and key[0] not in params:   # views of one tensor: stored once
            seen.add(key)
            total += storage.nbytes()
        keep.append(t)

    def unpack(_):          # never called: the graph is not backpropagated
        raise RuntimeError("_count_saved_bytes graphs are not meant for backward")

    from torch.nn.attention import SDPBackend, sdpa_kernel

    x = torch.zeros(1, in_chans, img_size, img_size)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION), \
                torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            model(x).sum()
    except RuntimeError:            # e.g. an attention mask the flash kernel rejects
        seen.clear()
        keep.clear()
        total = 0
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            model(x).sum()
    keep.clear()
    return total


@lru_cache(maxsize=64)
def saved_activation_gb(name: str, in_chans: int = 3, num_classes: int = 1000,
                        img_size: int = 224) -> float:
    """GB that autograd saves for the backward pass of ONE ``img_size``² image (fp32).

    Everything the graph keeps per image (activations, attention maps, masks), without the
    parameters it saves by reference (those are read once per batch, not per image).
    Raises ``ValueError`` if timm does not know the model or it cannot run on CPU."""
    import timm

    if not timm.is_model(name):
        raise ValueError(f"unknown timm model: {name}")
    model = _build(name, "cpu", in_chans, num_classes, None).train()
    try:
        return _count_saved_bytes(model, in_chans, img_size) / 1e9
    except (AssertionError, RuntimeError, ValueError) as first:
        # Fixed-resolution models (ViT/Swin at 224) reject other sizes: rebuild them.
        try:
            model = _build(name, "cpu", in_chans, num_classes, img_size).train()
            return _count_saved_bytes(model, in_chans, img_size) / 1e9
        except Exception as exc:  # timm raises TypeError for models without img_size
            raise ValueError(f"could not measure {name} at {img_size} px: {first} | {exc}") from exc
