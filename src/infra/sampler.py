"""Data sampling infrastructure."""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def make_weighted_sampler(labels: List[int]) -> torch.utils.data.WeightedRandomSampler:
    counts = np.bincount(np.asarray(labels, dtype=np.int64))
    counts = np.clip(counts, 1, None)
    class_weights = 1.0 / counts
    sample_weights = [float(class_weights[label_id]) for label_id in labels]
    return torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )
