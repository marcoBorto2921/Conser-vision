"""
src/models/model.py
--------------------
Image classifier built on top of timm pretrained backbones.
Supports EfficientNet, ConvNeXt, ViT and anything available in timm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models import create_model


class WildlifeClassifier(nn.Module):
    """Wildlife species classifier with a timm backbone.

    Args:
        model_name: timm model identifier (e.g. 'efficientnet_b3', 'convnext_base').
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        dropout: Dropout probability before the classifier head.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Load backbone without classifier head
        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,           # remove default head
            global_pool="avg",
        )

        # Get feature dimension
        num_features: int = self.backbone.num_features

        # Custom head: BN → Dropout → Linear
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (no softmax).

        Shape: (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def get_optimizer_param_groups(
        self,
        lr: float = 1e-4,
        lr_backbone_multiplier: float = 0.1,
    ) -> list[dict]:
        """Return parameter groups with differential learning rates.

        The backbone uses a smaller lr to preserve pretrained features.
        The head uses the full lr.
        """
        return [
            {
                "params": self.backbone.parameters(),
                "lr": lr * lr_backbone_multiplier,
                "name": "backbone",
            },
            {
                "params": self.head.parameters(),
                "lr": lr,
                "name": "head",
            },
        ]


def build_model(
    model_name: str,
    num_classes: int = 8,
    pretrained: bool = True,
    dropout: float = 0.3,
    checkpoint_path: str | None = None,
) -> WildlifeClassifier:
    """Instantiate a WildlifeClassifier, optionally loading from checkpoint."""
    model = WildlifeClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Handle state dicts saved with or without 'model_state_dict' key
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {checkpoint_path}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model_name} | Params: {n_params:.1f}M | Classes: {num_classes}")

    return model
