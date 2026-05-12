import timm
import torch
import torch.nn as nn
import torch.optim


class BigEarthViT(nn.Module):
    """Vision Transformer for multi-label classification on BigEarthNet-S2.

    Wraps a pretrained ViT from timm and replaces the classification head
    with one suited for multi-label output (19 classes, no softmax).
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 19,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            model_name: timm model identifier.
            num_classes: Number of output classes (19 for BigEarthNet).
            pretrained: Whether to load ImageNet pretrained weights.
            dropout: Dropout rate before the classification head.
        """
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove the original classification head
        )

        embed_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W).
        Returns:
            Logits of shape (B, num_classes). No sigmoid applied —
            use BCEWithLogitsLoss during training.
        """
        features = self.backbone(x)
        return self.head(features)


def build_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 19,
    pretrained: bool = True,
) -> BigEarthViT:
    """Instantiate and return the model."""
    model = BigEarthViT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    return model


def build_llrd_optimizer(
    model: BigEarthViT,
    lr_base: float,
    weight_decay: float,
    llrd_decay: float = 0.75,
) -> torch.optim.AdamW:
    """Build AdamW with layer-wise learning rate decay (LLRD) for ViT fine-tuning.

    LR decays geometrically from head (lr_base) toward patch_embed
    (lr_base * llrd_decay^(num_blocks+2)). Bias and norm params skip weight decay.

    Decay schedule (ViT-B/16, 12 blocks):
      head            → lr_base
      backbone.norm   → lr_base * decay^1
      blocks[11]      → lr_base * decay^2
      ...
      blocks[0]       → lr_base * decay^13
      patch_embed     → lr_base * decay^14
    """
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
    num_blocks = len(model.backbone.blocks)

    def _add(params_decay, params_no_decay, lr):
        if params_decay:
            groups.append({"params": params_decay, "lr": lr, "weight_decay": weight_decay})
        if params_no_decay:
            groups.append({"params": params_no_decay, "lr": lr, "weight_decay": 0.0})

    # Head — full lr_base
    _add(*_split(model.head.named_parameters()), lr_base)

    # backbone top-level params (norm, cls_token, pos_embed) — depth 1
    top_named = [
        (n, p) for n, p in model.backbone.named_parameters()
        if "blocks" not in n and "patch_embed" not in n
    ]
    _add(*_split(top_named), lr_base * llrd_decay)

    # Transformer blocks — deeper blocks get lower LR
    for block_idx in reversed(range(num_blocks)):
        depth = num_blocks - block_idx  # block 11 → depth 1, block 0 → depth 12
        lr_block = lr_base * (llrd_decay ** depth)
        _add(*_split(model.backbone.blocks[block_idx].named_parameters()), lr_block)

    # patch_embed — deepest
    _add(*_split(model.backbone.patch_embed.named_parameters()), lr_base * (llrd_decay ** (num_blocks + 2)))

    return torch.optim.AdamW(groups)
