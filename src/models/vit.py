import timm
import torch
import torch.nn as nn


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
