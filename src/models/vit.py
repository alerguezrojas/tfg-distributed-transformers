import timm
import torch
import torch.nn as nn
import torch.optim

from src.models.backbones import available_backbones, resolve_backbone

DEFAULT_IMG_SIZE = 224

_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Print an '[aviso]' line the first time ``key`` is seen in this process."""
    if key not in _warned:
        _warned.add(key)
        print(f"  [aviso] {message}")


class TimmClassifier(nn.Module):
    """Any timm backbone + a ``Dropout → Linear`` classification head.

    The head returns raw logits (no sigmoid/softmax): the task decides how to read
    them. ``num_classes`` and ``in_chans`` come from the dataset spec (BigEarthNet:
    19 classes, 3 channels). Supports ViT, DeiT, Swin, ResNet, EfficientNet and any
    other timm model; with ``in_chans != 3`` timm adapts the pretrained stem.
    A name in the backbone registry or ``hf:<repo>`` builds a non-timm encoder instead
    (:mod:`src.models.backbones`); any other name is timm, built exactly as always.
    ``freeze_backbone`` trains only the head (linear probe).
    ``img_size`` (e.g. 512) builds the backbone for that input side; timm resizes
    the pretrained position embeddings. ``None`` keeps the model's default (224).
    """

    # Name prefixes read as "a ViT" by ``is_vit`` (model parallelism splits ``.blocks``).
    # LLRD does not use them: see ``supports_llrd``.
    VIT_PATTERNS = ("vit_", "deit_", "swin_")

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 19,
        pretrained: bool = True,
        dropout: float = 0.1,
        in_chans: int = 3,
        img_size: int | None = None,
        pooling: str | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.in_chans = in_chans
        self.freeze_backbone = freeze_backbone
        self.backbone_spec = resolve_backbone(model_name, pooling)   # None → timm
        spec = self.backbone_spec
        self.img_size = img_size or (spec.default_img_size if spec else DEFAULT_IMG_SIZE)
        if spec is not None:
            self._warn_about(spec)

        kwargs = {"pretrained": pretrained, "num_classes": 0,  # drop timm's own head
                  "in_chans": in_chans}
        if spec is not None:
            self.backbone = spec.build({"pretrained": pretrained, "in_chans": in_chans,
                                        "img_size": img_size, "pooling": spec.pooling})
        else:
            try:
                self.backbone = self._create_timm(model_name, img_size, kwargs)
            except RuntimeError as exc:
                if "Unknown model" not in str(exc):
                    raise
                # Not timm's either: a misspelt registered name must not fail silently.
                raise ValueError(
                    f"model.name {model_name!r} is not a timm model, a registered backbone "
                    f"{available_backbones()} or an 'hf:<repo>' name") from exc

        embed_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        if freeze_backbone:   # linear probe: only the head learns
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    @staticmethod
    def _create_timm(model_name: str, img_size: int | None, kwargs: dict) -> nn.Module:
        if not img_size:
            return timm.create_model(model_name, **kwargs)
        try:
            return timm.create_model(model_name, img_size=img_size, **kwargs)
        except TypeError:
            # CNN families take no img_size kwarg: they accept any input side.
            return timm.create_model(model_name, **kwargs)

    @staticmethod
    def _warn_about(spec) -> None:
        if spec.non_commercial:
            _warn_once(f"license:{spec.name}",
                       f"'{spec.name}' tiene licencia {spec.license} (solo uso no comercial): "
                       "no usar los pesos ni los resultados con fines comerciales.")
        if not spec.imagenet_normalized:
            _warn_once(f"norm:{spec.name}",
                       f"'{spec.name}' se entrenó con mean={spec.mean} std={spec.std}, pero los "
                       "transforms del dataset normalizan con la media/desviación de ImageNet.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:   # frozen features: no dropout / BatchNorm statistics drift
            self.backbone.eval()
        return self

    @property
    def supports_llrd(self) -> bool:
        """True when ``build_llrd_optimizer`` can split the backbone into layers: a
        registered/``hf:`` encoder with ``llrd_layout()``, a timm ViT (``blocks`` +
        ``patch_embed``) or any timm model with ``group_matcher()`` (Swin, ConvNeXt,
        EfficientNet, ResNet, EVA…)."""
        if self.backbone_spec is not None:
            return self.backbone_spec.llrd
        b = self.backbone
        return (hasattr(b, "llrd_layout") or (hasattr(b, "blocks") and hasattr(b, "patch_embed"))
                or hasattr(b, "group_matcher"))

    @property
    def is_vit(self) -> bool:
        """True for a ViT-like backbone by name (or a registered/``hf:`` encoder with
        ``llrd_layout()``). Only model parallelism uses it (it splits ``backbone.blocks``);
        whether LLRD applies is ``supports_llrd``."""
        if self.backbone_spec is not None:
            return self.backbone_spec.llrd
        return any(self.model_name.startswith(p) for p in self.VIT_PATTERNS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, in_chans, H, W).
        Returns:
            Logits of shape (B, num_classes). No activation applied — the
            task's loss works on logits.
        """
        features = self.backbone(x)
        return self.head(features)


def input_channels(model: nn.Module, default: int = 3) -> int:
    """Input channels of a (possibly DDP- or pipeline-wrapped) TimmClassifier."""
    for attr in ("module", "base"):          # DistributedDataParallel / ModelParallelViT
        model = getattr(model, attr, model)
    return getattr(model, "in_chans", default)


def input_size(model: nn.Module, default: int = DEFAULT_IMG_SIZE) -> int:
    """Input side (pixels) of a (possibly DDP- or pipeline-wrapped) TimmClassifier."""
    for attr in ("module", "base"):          # DistributedDataParallel / ModelParallelViT
        model = getattr(model, attr, model)
    return getattr(model, "img_size", default)


# Backward-compatible aliases (names used before multi-dataset support)
BigEarthModel = TimmClassifier
BigEarthViT = TimmClassifier


def build_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 19,
    pretrained: bool = True,
    in_chans: int = 3,
    dropout: float = 0.1,
    img_size: int | None = None,
    pooling: str | None = None,
    freeze_backbone: bool = False,
) -> TimmClassifier:
    """Instantiate and return the model (``dropout`` = rate before the head)."""
    return TimmClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        in_chans=in_chans,
        img_size=img_size,
        pooling=pooling,
        freeze_backbone=freeze_backbone,
    )


def _llrd_layout(backbone: nn.Module):
    """``(embedding, [block…], other)`` parameter lists as ``(name, param)`` pairs."""
    if hasattr(backbone, "llrd_layout"):      # non-timm encoders (src.models.backbones)
        return backbone.llrd_layout()
    if _is_vit_layout(backbone):                                                # timm ViT/DeiT
        # "top" is every parameter outside patch_embed and blocks, by identity: a name
        # filter would also drop CaiT's ``blocks_token_only`` or XCiT's ``cls_attn_blocks``.
        seen = {id(p) for p in backbone.patch_embed.parameters()}
        seen |= {id(p) for p in backbone.blocks.parameters()}
        top = [(n, p) for n, p in backbone.named_parameters() if id(p) not in seen]
        return (list(backbone.patch_embed.named_parameters()),
                [list(b.named_parameters()) for b in backbone.blocks], top)
    return _group_matcher_layout(backbone)


def _is_vit_layout(backbone: nn.Module) -> bool:
    return (not hasattr(backbone, "llrd_layout") and hasattr(backbone, "blocks")
            and hasattr(backbone, "patch_embed"))


#: Depth the decay of a non-ViT model is spread over (a ViT-S/B has 12 blocks): with it,
#: ``llrd_decay`` means the same total decay from head to stem for every family.
LLRD_REFERENCE_BLOCKS = 12


def _group_matcher_layout(backbone: nn.Module):
    """The same ``(embedding, [block…], other)`` split for any timm model, from its own
    fine-grained ``group_matcher()`` (the grouping timm uses for layer decay): the first
    group is the stem / patch embedding and every later group is one block, in
    input-to-output order (timm folds the final norm into the last block). "other" holds
    what is left of timm's classifier ``head`` module (``head.norm``…), which timm leaves
    unmatched in a last group of its own: like a ViT's final norm, it gets ``decay^1``."""
    from timm.models import group_parameters
    if not hasattr(backbone, "group_matcher"):
        raise ValueError(f"LLRD needs a backbone with blocks or group_matcher(), "
                         f"got {type(backbone).__name__}: set training.llrd_decay: 0.0")
    named = dict(backbone.named_parameters())
    layers = group_parameters(backbone, backbone.group_matcher(coarse=False))
    # what is left of timm's classifier ``head`` module (``head.norm``, ``head.pre_logits``
    # in ConvNeXt/MaxViT…) is not a block: it goes to "other", next to the head.
    top = [(n, named[n]) for k in sorted(layers) for n in layers[k] if n.startswith("head.")]
    groups = [g for g in ([(n, named[n]) for n in layers[k] if not n.startswith("head.")]
                          for k in sorted(layers)) if g]
    if len(groups) < 2:
        raise ValueError(f"{type(backbone).__name__}.group_matcher() gives a single group; "
                         "LLRD needs a stem and blocks: set training.llrd_decay: 0.0")
    return groups[0], groups[1:], top


def build_llrd_optimizer(
    model: TimmClassifier,
    lr_base: float,
    weight_decay: float,
    llrd_decay: float = 0.75,
) -> torch.optim.AdamW:
    """Build AdamW with layer-wise learning rate decay (LLRD) for ViT fine-tuning.

    LR decays geometrically from head (lr_base) toward patch_embed
    (lr_base * llrd_decay^(num_blocks+2)). Biases, norms and every 1-D parameter
    (BatchNorm, layer-scale) skip weight decay.

    Valid for any backbone with ``model.supports_llrd``: timm ViT/DeiT use their
    ``blocks``; other timm models (Swin, ConvNeXt, EfficientNet, ResNet…) are split by
    their ``group_matcher()``, so "block" there is a timm block (a stage's block). Their
    depth varies a lot (5 groups in MaxViT-T, 37 in ConvNeXt-S), so for them the decay is
    spread as if they had ``LLRD_REFERENCE_BLOCKS`` (12) blocks: the first block gets
    ``decay^12`` and the stem ``decay^14``, as in a ViT-S/B.

    Decay schedule (ViT-B/16, 12 blocks):
      head                  → lr_base
      backbone.norm, tokens → lr_base * decay^1
      blocks[11]            → lr_base * decay^1
      ...
      blocks[0]             → lr_base * decay^12
      patch_embed           → lr_base * decay^14
    """
    if not model.supports_llrd:
        raise ValueError(
            f"LLRD cannot split the backbone of '{model.model_name}' into layers: "
            "set training.llrd_decay: 0.0 (plain AdamW)."
        )

    no_decay = {"bias", "norm"}

    def _split(named_params):
        decay_p, no_decay_p = [], []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or any(kw in name for kw in no_decay):
                no_decay_p.append(param)
            else:
                decay_p.append(param)
        return decay_p, no_decay_p

    groups: list[dict] = []
    embed_named, blocks_named, top_named = _llrd_layout(model.backbone)
    num_blocks = len(blocks_named)
    vit = _is_vit_layout(model.backbone) or hasattr(model.backbone, "llrd_layout")
    # exponent of the decay at a block's depth (1 = last block) and at the stem
    scale = 1.0 if vit else LLRD_REFERENCE_BLOCKS / num_blocks
    stem_exp = num_blocks + 2 if vit else LLRD_REFERENCE_BLOCKS + 2

    def _add(params_decay, params_no_decay, lr):
        if params_decay:
            groups.append({"params": params_decay, "lr": lr, "weight_decay": weight_decay})
        if params_no_decay:
            groups.append({"params": params_no_decay, "lr": lr, "weight_decay": 0.0})

    # Head — full lr_base
    _add(*_split(model.head.named_parameters()), lr_base)

    # backbone top-level params (norm, cls_token, pos_embed) — depth 1
    _add(*_split(top_named), lr_base * llrd_decay)

    # Transformer blocks — deeper blocks get lower LR
    for block_idx in reversed(range(num_blocks)):
        depth = num_blocks - block_idx  # block 11 → depth 1, block 0 → depth 12
        lr_block = lr_base * (llrd_decay ** (depth * scale))
        _add(*_split(blocks_named[block_idx]), lr_block)

    # patch_embed — deepest
    _add(*_split(embed_named), lr_base * (llrd_decay ** stem_exp))

    grouped = [id(p) for g in groups for p in g["params"]]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if len(grouped) != len(set(grouped)) or set(grouped) != trainable:
        raise RuntimeError(f"LLRD layout of '{model.model_name}' misses or repeats parameters "
                           f"({len(set(grouped))} grouped of {len(trainable)} trainable)")
    return torch.optim.AdamW(groups)


def warn_if_below_lr_min(optimizer: torch.optim.Optimizer, lr_min: float) -> int:
    """Number of parameter groups that start below the cosine's ``eta_min`` (it would
    *raise* their LR during training, flattening LLRD), with an ``[aviso]`` if any."""
    low = [g["lr"] for g in optimizer.param_groups if g["lr"] < lr_min]
    if low:
        print(f"  [aviso] {len(low)} de {len(optimizer.param_groups)} grupos de LLRD empiezan "
              f"por debajo de lr_min={lr_min:g} (mínimo {min(low):.1e}): el coseno les subirá "
              "la tasa. Bajen lr_min o suban llrd_decay.")
    return len(low)
