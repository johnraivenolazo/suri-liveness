"""Training use cases: Model training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    accuracy: float
    recall_per_class: np.ndarray


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        _, predicted = outputs.max(1)
        total += int(labels.size(0))
        correct += int(predicted.eq(labels).sum().item())

    return running_loss / max(len(loader), 1), (100.0 * correct / max(total, 1))


def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> EvalMetrics:
    model.eval()

    running_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += float(loss.item())

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)

    acc = accuracy_score(y_true, y_pred) * 100.0 if y_true.size else 0.0
    recall = recall_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    return EvalMetrics(
        loss=running_loss / max(len(loader), 1),
        accuracy=acc,
        recall_per_class=recall,
    )
