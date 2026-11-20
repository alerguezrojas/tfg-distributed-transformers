import timm
import torch
import torch.nn as nn
import torch.optim

from src.models.backbones import resolve_backbone

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

    # Model families whose backbone exposes `.blocks` and `.patch_embed` → LLRD-compatible
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
        elif img_size:
            try:
                self.backbone = timm.create_model(model_name, img_size=img_size, **kwargs)
            except TypeError:
                # CNN families take no img_size kwarg: they accept any input side.
                self.backbone = timm.create_model(model_name, **kwargs)
        else:
            self.backbone = timm.create_model(model_name, **kwargs)

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
    def is_vit(self) -> bool:
        """True for backbones LLRD can walk: timm ViT/DeiT/Swin, or a registered/``hf:``
        encoder that exposes ``llrd_layout()``."""
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
    top = [(n, p) for n, p in backbone.named_parameters()
           if "blocks" not in n and "patch_embed" not in n]
    return (list(backbone.patch_embed.named_parameters()),
            [list(b.named_parameters()) for b in backbone.blocks], top)


def build_llrd_optimizer(
    model: TimmClassifier,
    lr_base: float,
    weight_decay: float,
    llrd_decay: float = 0.75,
) -> torch.optim.AdamW:
    """Build AdamW with layer-wise learning rate decay (LLRD) for ViT fine-tuning.

    LR decays geometrically from head (lr_base) toward patch_embed
    (lr_base * llrd_decay^(num_blocks+2)). Bias and norm params skip weight decay.

    Only valid for ViT/DeiT/Swin backbones (model.is_vit == True).
    For CNN models, use standard AdamW instead.

    Decay schedule (ViT-B/16, 12 blocks):
      head            → lr_base
      backbone.norm   → lr_base * decay^1
      blocks[11]      → lr_base * decay^2
      ...
      blocks[0]       → lr_base * decay^13
      patch_embed     → lr_base * decay^14
    """
    if not model.is_vit:
        raise ValueError(
            f"LLRD requires a ViT/DeiT/Swin backbone, got '{model.model_name}'. "
            "Use standard AdamW for CNN models."
        )

    no_decay = {"bias", "norm"}

    def _split(named_params):
        decay_p, no_decay_p = [], []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if any(kw in name for kw in no_decay):
                no_decay_p.append(param)
            else:
                decay_p.append(param)
        return decay_p, no_decay_p

    groups: list[dict] = []
    embed_named, blocks_named, top_named = _llrd_layout(model.backbone)
    num_blocks = len(blocks_named)

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
        lr_block = lr_base * (llrd_decay ** depth)
        _add(*_split(blocks_named[block_idx]), lr_block)

    # patch_embed — deepest
    _add(*_split(embed_named), lr_base * (llrd_decay ** (num_blocks + 2)))

    return torch.optim.AdamW(groups)
