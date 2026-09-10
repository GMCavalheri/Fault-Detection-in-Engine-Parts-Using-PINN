"""Helpers for feeding CWRU windows into PyTorch models."""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.physics.bearing import expected_fault_frequency_hz


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


class PhysicsWindowDataset(Dataset):
    """Like WindowDataset, plus a physics-informed regression target: the
    bearing defect frequency predicted by src/physics/bearing.py from each
    window's (label, RPM). `Normal` windows (no defect frequency) get
    target=0 and mask=0, so the loss simply skips them - see
    src/models/physics_cnn.py::train_physics_cnn.

    `freq_scale_hz` normalizes the Hz-scale target to roughly O(1), so its
    MSE loss is a comparable magnitude to the cross-entropy classification
    loss. 200.0 covers this dataset's whole range (defect frequencies here
    span ~68-162 Hz across all classes and CWRU's 1730-1797 RPM loads).
    """

    def __init__(
        self,
        windows: np.ndarray,
        class_targets: np.ndarray,
        labels: np.ndarray,
        rpms: np.ndarray,
        mean: float,
        std: float,
        freq_scale_hz: float = 200.0,
    ):
        self.windows = ((windows - mean) / std).astype(np.float32)
        self.class_targets = class_targets
        self.freq_scale_hz = freq_scale_hz

        physics_targets, physics_masks = [], []
        for label, rpm in zip(labels, rpms):
            freq_hz = expected_fault_frequency_hz(label, rpm)
            if freq_hz is None:
                physics_targets.append(0.0)
                physics_masks.append(0.0)
            else:
                physics_targets.append(freq_hz / freq_scale_hz)
                physics_masks.append(1.0)
        self.physics_targets = np.array(physics_targets, dtype=np.float32)
        self.physics_masks = np.array(physics_masks, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx]).unsqueeze(0)  # (1, window_size)
        y = self.class_targets[idx]
        physics_target = torch.tensor(self.physics_targets[idx], dtype=torch.float32)
        physics_mask = torch.tensor(self.physics_masks[idx], dtype=torch.float32)
        return x, y, physics_target, physics_mask
