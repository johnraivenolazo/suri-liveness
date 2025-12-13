from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import timm


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    num_classes: int = 3
    pretrained: bool = False


def _replace_classifier_head(model: nn.Module, num_classes: int) -> None:
    if hasattr(model, "classifier") and getattr(model, "classifier") is not None:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "head") and getattr(model, "head") is not None:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "fc") and getattr(model, "fc") is not None:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return

    raise ValueError("Could not locate classifier head on model")


def create_model(cfg: ModelConfig, *, device: Optional[torch.device] = None) -> nn.Module:
    model = timm.create_model(cfg.model_name, pretrained=cfg.pretrained)
    _replace_classifier_head(model, cfg.num_classes)

    if device is not None:
        model = model.to(device)

    return model


def freeze_backbone(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

    for attr in ("classifier", "head", "fc"):
        head = getattr(model, attr, None)
        if head is None:
            continue
        for p in head.parameters():
            p.requires_grad = True
        return

    raise ValueError("Could not locate classifier head to unfreeze")
