"""Helpers for feeding C-MAPSS sequences into PyTorch models."""

import numpy as np
import torch
from torch.utils.data import Dataset


class RulSequenceDataset(Dataset):
    """Sliding-window sequences, standardized per-feature with train-set
    statistics (mirrors src/data/torch_utils.py::WindowDataset's
    global-standardization approach, just per-feature instead of scalar
    since each sensor has a different scale).
    """

    def __init__(self, sequences: np.ndarray, targets: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.sequences = ((sequences - mean) / std).astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.targets[idx])


def compute_sequence_train_stats(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sensor mean/std over all training windows and timesteps."""
    flat = sequences.reshape(-1, sequences.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # guard against a residual near-constant feature
    return mean, std
