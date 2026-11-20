"""BackboneSpec — everything the framework needs to know about a non-timm backbone.

Same idea as :class:`~src.data.spec.DatasetSpec`: a backbone that does not come from
timm is described once by a :class:`BackboneSpec` and registered by name, so adding one
means writing a spec — not touching the training loop. The config picks the backbone
with ``model.name``, which is resolved like this (see :func:`resolve_backbone`):

* a name in the registry → its spec (e.g. ``ijepa_vith14_1k``);
* ``hf:<repo>`` → the generic Hugging Face ``transformers`` adapter (``AutoModel``);
* anything else → ``None``: it is a timm model, built exactly as before.

``transformers`` is an optional dependency (``pip install '.[hf]'``) imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import torch
import torch.nn as nn

SOURCES = ("timm", "transformers", "torch_hub")
POOLINGS = ("cls", "mean")
HF_PREFIX = "hf:"

# What the dataset transforms apply today (torchvision Normalize for RGB inputs).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: build(cfg) → encoder. ``cfg`` keys: pretrained, in_chans, img_size, pooling.
#: The encoder maps (B, C, H, W) → (B, D) and exposes ``num_features`` = D. When the
#: spec sets ``llrd`` it also provides ``llrd_layout()`` (see ``build_llrd_optimizer``).
BackboneFactory = Callable[[dict], nn.Module]


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    source: str                          # 'timm' | 'transformers' | 'torch_hub'
    build: BackboneFactory
    description: str = ""
    pooling: str = "mean"                # how tokens become one vector: 'cls' | 'mean'
    patch_size: int | None = None
    default_img_size: int = 224
    mean: tuple[float, ...] = IMAGENET_MEAN   # normalization the backbone was trained with
    std: tuple[float, ...] = IMAGENET_STD
    llrd: bool = False                   # encoder exposes llrd_layout() (layer-wise LR decay)
    license: str = ""                    # e.g. 'CC BY-NC 4.0'
    non_commercial: bool = False

    def __post_init__(self):
        if not self.name or (self.name != self.name.lower() and not self.name.startswith(HF_PREFIX)):
            raise ValueError(f"backbone name {self.name!r} must be lowercase")
        if self.source not in SOURCES:
            raise ValueError(f"{self.name}: unknown source {self.source!r}; available: {SOURCES}")
        if self.pooling not in POOLINGS:
            raise ValueError(f"{self.name}: pooling must be one of {POOLINGS}, got {self.pooling!r}")
        if len(self.mean) != len(self.std):
            raise ValueError(f"{self.name}: mean and std must have the same length")

    @property
    def imagenet_normalized(self) -> bool:
        return tuple(self.mean) == IMAGENET_MEAN and tuple(self.std) == IMAGENET_STD


# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, BackboneSpec] = {}


def register_backbone(spec: BackboneSpec) -> BackboneSpec:
    """Add ``spec`` to the registry (idempotent for the same object)."""
    current = _REGISTRY.get(spec.name)
    if current is not None and current is not spec:
        raise ValueError(f"backbone {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def available_backbones() -> list[str]:
    return sorted(_REGISTRY)


def is_hf_name(name: str) -> bool:
    return name.startswith(HF_PREFIX)


def resolve_backbone(name: str, pooling: str | None = None) -> BackboneSpec | None:
    """Spec for ``model.name``; ``None`` means "a timm model" (the historical path).

    ``pooling`` (``model.pooling``) overrides the spec's default for registered and
    ``hf:`` backbones.
    """
    spec = _REGISTRY.get(name)
    if spec is None and is_hf_name(name):
        repo = name[len(HF_PREFIX):]
        if not repo:
            raise ValueError("model.name 'hf:' needs a repository, e.g. 'hf:facebook/ijepa_vith14_1k'")
        spec = BackboneSpec(
            name=name, source="transformers", build=lambda cfg, _r=repo: build_hf_encoder(_r, cfg),
            description=f"Hugging Face transformers AutoModel: {repo}", pooling="mean", llrd=True)
    if spec is not None and pooling is not None and pooling != spec.pooling:
        spec = replace(spec, pooling=pooling)
    return spec


# ── Hugging Face transformers adapter ────────────────────────────────────────

class HFEncoder(nn.Module):
    """``transformers`` ViT-like model → (B, D) feature vector.

    Pools ``last_hidden_state`` by the mean of the tokens (what I-JEPA uses: it has no
    [CLS]) or by its first token (``pooling='cls'``). ``llrd_layout()`` needs the usual
    ``embeddings`` + ``encoder.layer`` structure.
    """

    def __init__(self, model: nn.Module, pooling: str = "mean"):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        self.model = model
        self.pooling = pooling
        self.num_features = int(model.config.hidden_size)
        self._native_size = getattr(model.config, "image_size", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self._native_size is not None and x.shape[-1] != self._native_size:
            kwargs["interpolate_pos_encoding"] = True   # other input side: resize pos. embeddings
        tokens = self.model(pixel_values=x, **kwargs).last_hidden_state
        return tokens.mean(dim=1) if self.pooling == "mean" else tokens[:, 0]

    def llrd_layout(self):
        """``(embedding params, [block params…], other params)`` as ``(name, param)`` lists."""
        embed = getattr(self.model, "embeddings", None)
        layers = _find_layers(self.model)
        if embed is None or layers is None:
            raise ValueError("LLRD needs a model with 'embeddings' and a list of layers "
                             "(encoder.layer / encoder.layers / layers); "
                             "set training.llrd_decay: 0.0 for this backbone")

        def named(module):   # lowercase: LLRD skips weight decay for names with 'norm'
            return [(n.lower(), p) for n, p in module.named_parameters()]

        inside = {id(p) for m in (embed, layers) for p in m.parameters()}
        rest = [(n.lower(), p) for n, p in self.model.named_parameters() if id(p) not in inside]
        return named(embed), [named(b) for b in layers], rest


def _find_layers(model: nn.Module):
    """The transformer blocks of a ``transformers`` model: ``encoder.layer`` (BERT/ViT style,
    transformers 4.x), ``encoder.layers`` or ``layers`` (I-JEPA in transformers 5.x)."""
    holders = (getattr(model, "encoder", None), model)
    for holder in holders:
        for attr in ("layer", "layers"):
            layers = getattr(holder, attr, None)
            if isinstance(layers, nn.ModuleList):
                return layers
    return None


def build_hf_encoder(repo: str, cfg: dict, config=None) -> HFEncoder:
    """Load ``repo`` with ``AutoModel`` (random weights from its config if not pretrained).

    ``config`` (a ``transformers`` config object) replaces the one downloaded from the hub
    when ``pretrained`` is false: lets tests build a tiny model without network.
    """
    if cfg.get("in_chans", 3) != 3:
        raise ValueError(f"'hf:{repo}' expects 3-channel images, the dataset has "
                         f"{cfg['in_chans']}; pick a 3-channel dataset or a timm model")
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            f"The backbone '{repo}' needs the optional 'transformers' package: install it "
            "with  pip install 'tfg-distributed-transformers[hf]'  (uv sync --extra hf)") from exc
    if cfg.get("pretrained", True):
        model = AutoModel.from_pretrained(repo)
    else:
        model = AutoModel.from_config(config if config is not None
                                      else AutoConfig.from_pretrained(repo))
    return HFEncoder(model, pooling=cfg.get("pooling") or "mean")


# ── Registered backbones ─────────────────────────────────────────────────────

_IJEPA_NOTE = ("pesos CC BY-NC 4.0 (solo uso no comercial); sin [CLS]: media de tokens; "
               "normalización de ImageNet")


def _ijepa(name: str, repo: str, description: str) -> BackboneSpec:
    return register_backbone(BackboneSpec(
        name=name, source="transformers", build=lambda cfg, _r=repo: build_hf_encoder(_r, cfg),
        description=f"{description}; {_IJEPA_NOTE}", pooling="mean", patch_size=14,
        default_img_size=224, llrd=True, license="CC BY-NC 4.0", non_commercial=True))


_ijepa("ijepa_vith14_1k", "facebook/ijepa_vith14_1k", "I-JEPA ViT-H/14, ImageNet-1k, 224 px")
_ijepa("ijepa_vith14_22k", "facebook/ijepa_vith14_22k", "I-JEPA ViT-H/14, ImageNet-22k, 224 px")
