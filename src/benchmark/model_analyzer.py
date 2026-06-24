"""ModelAnalyzer — FLOPs, parameters and static memory of a model."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.benchmark.value_objects import ModelInfo


class ModelAnalyzer:
    def __init__(self, model: nn.Module, model_name: str, device: torch.device):
        self._model = model
        self._name = model_name
        self._device = device

    def analyze(self) -> ModelInfo:
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        weight_mb = total * 4 / 1e6
        gradient_mb = trainable * 4 / 1e6
        optimizer_mb = trainable * 8 / 1e6
        flops, activation_mb = self._run_summary()
        return ModelInfo(
            name=self._name,
            total_params=total,
            trainable_params=trainable,
            flops_per_image_mflops=flops,
            weight_mb=weight_mb,
            gradient_mb=gradient_mb,
            optimizer_mb=optimizer_mb,
            activation_mb_per_image=activation_mb,
        )

    def _run_summary(self) -> tuple[float, float]:
        try:
            from torchinfo import summary
            stats = summary(self._model, input_size=(1, 3, 224, 224),
                            verbose=0, device=torch.device("cpu"))
            return stats.total_mult_adds / 1e6, getattr(stats, "total_output_bytes", 0) / 1e6
        except Exception:
            return 0.0, 0.0
