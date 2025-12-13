"""Core layer: Business logic and entities."""

from src.core.labels import LabelSpec, infer_label_spec
from src.core.models import ModelConfig, create_model, freeze_backbone

__all__ = [
    "LabelSpec",
    "infer_label_spec",
    "ModelConfig",
    "create_model",
    "freeze_backbone",
]
