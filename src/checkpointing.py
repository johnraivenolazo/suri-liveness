from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TrainState:
    epoch: int
    best_metric: float


def save_state(
    *,
    path: str,
    epoch: int,
    best_metric: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> None:
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, path)


def load_state(
    *,
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[str] = None,
) -> TrainState:
    state = torch.load(path, map_location=map_location)

    model.load_state_dict(state["model_state_dict"], strict=True)

    if optimizer is not None and "optimizer_state_dict" in state and state["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return TrainState(epoch=int(state.get("epoch", 0)), best_metric=float(state.get("best_metric", 0.0)))


def find_epoch_checkpoints(save_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(save_dir, "epoch*_metric*.pth")))


def best_epoch_from_filenames(files: list[str]) -> int:
    epoch_pattern = re.compile(r"epoch(\d+)_metric")
    epochs = []
    for f in files:
        m = epoch_pattern.search(os.path.basename(f))
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else 0
