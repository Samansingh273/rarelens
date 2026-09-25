"""ResNet-18 / EfficientNet-B0 classifiers with ImageNet transfer learning."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_classifier(arch: str = "resnet18", num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)  # new 7-way head, rest fine-tuned
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return m
