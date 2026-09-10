"""Helpers for feeding CWRU windows into PyTorch models."""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Raw signal windows as (1, window_size) tensors, globally standardized.

    Standardizing with train-set statistics (rather than per-window) preserves
    relative amplitude differences between classes, which is diagnostic for
    fault severity - per-window normalization would erase it.
    """

    def __init__(self, windows: np.ndarray, targets: np.ndarray, mean: float, std: float):
        self.windows = ((windows - mean) / std).astype(np.float32)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx]).unsqueeze(0)  # (1, window_size)
        y = self.targets[idx]
        return x, y


def compute_train_stats(windows: np.ndarray) -> tuple[float, float]:
    return float(windows.mean()), float(windows.std())
