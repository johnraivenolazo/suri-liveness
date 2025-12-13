from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from labels import LabelSpec


@dataclass(frozen=True)
class DataPaths:
    data_root: str
    train_json: str
    val_json: str


class JsonImageDataset(Dataset):
    def __init__(
        self,
        *,
        root_dir: str,
        json_path: str,
        label_spec: LabelSpec,
        transform: Optional[Any] = None,
        strip_prefix: Optional[str] = None,
        mode: str = "train",
    ) -> None:
        self.root_dir = root_dir
        self.json_path = json_path
        self.label_spec = label_spec
        self.transform = transform
        self.strip_prefix = strip_prefix
        self.mode = mode

        try:
            meta = pd.read_json(json_path, orient="index")
        except ValueError:
            meta = pd.read_json(json_path)

        self.meta = meta
        self.img_paths = meta.index.tolist()

        # Keep raw labels as-is; LabelSpec decides how to interpret them.
        if label_spec.label_column not in meta.columns:
            raise ValueError(
                f"Label column {label_spec.label_column!r} not found in metadata columns: {list(meta.columns)[:20]}"
            )
        self.raw_labels = meta[label_spec.label_column].tolist()

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel_path = self.img_paths[idx]
        if not isinstance(rel_path, str):
            rel_path = str(rel_path)

        if self.strip_prefix and rel_path.startswith(self.strip_prefix):
            rel_path = rel_path[len(self.strip_prefix) :]

        img_full_path = os.path.join(self.root_dir, rel_path)

        try:
            image = Image.open(img_full_path).convert("RGB")
        except FileNotFoundError:
            # Try the next item to avoid hard failure on a single missing file.
            return self.__getitem__((idx + 1) % len(self))

        class_id = self.label_spec.to_class_id(self.raw_labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, class_id


def build_transforms(image_size: int) -> Tuple[Any, Any]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomAutocontrast(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x + (torch.randn_like(x) * 0.02) if torch.rand(1).item() < 0.3 else x
            ),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform
