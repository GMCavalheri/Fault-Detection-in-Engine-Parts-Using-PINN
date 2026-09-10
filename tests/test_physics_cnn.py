import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.torch_utils import PhysicsWindowDataset, compute_train_stats
from src.models.physics_cnn import PhysicsCnn1D, predict, train_physics_cnn
from src.physics.bearing import bpfi_hz

WINDOW_SIZE = 2048


def _synthetic_dataset(n_samples: int) -> PhysicsWindowDataset:
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(n_samples, WINDOW_SIZE))
    labels = np.array(["Normal", "IR007"] * (n_samples // 2))
    class_targets = np.array([0, 1] * (n_samples // 2))
    rpms = np.full(n_samples, 1797.0)
    mean, std = compute_train_stats(windows)
    return PhysicsWindowDataset(windows, class_targets, labels, rpms, mean, std)


def test_physics_cnn_forward_pass_shapes():
    model = PhysicsCnn1D(num_classes=2)
    x = torch.randn(4, 1, WINDOW_SIZE)
    class_logits, physics_pred = model(x)
    assert class_logits.shape == (4, 2)
    assert physics_pred.shape == (4,)


def test_physics_window_dataset_masks_normal_and_targets_fault():
    ds = _synthetic_dataset(n_samples=4)
    # index 0 is "Normal" -> mask 0, target 0
    _, _, target0, mask0 = ds[0]
    assert mask0.item() == 0.0
    assert target0.item() == 0.0
    # index 1 is "IR007" at 1797 RPM -> mask 1, target = bpfi(1797)/200
    _, _, target1, mask1 = ds[1]
    assert mask1.item() == 1.0
    assert target1.item() == pytest.approx(bpfi_hz(1797) / 200.0, rel=1e-4)


def test_train_physics_cnn_smoke_test_with_and_without_physics_loss():
    ds = _synthetic_dataset(n_samples=20)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    for physics_weight in (0.0, 1.0):
        model = PhysicsCnn1D(num_classes=2)
        history = train_physics_cnn(model, loader, loader, epochs=1, physics_weight=physics_weight)
        assert len(history) == 1
        assert set(history[0]) == {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}

        result = predict(model, loader)
        assert result["class_preds"].shape == (20,)
        assert result["physics_preds"].shape == (20,)
        assert result["physics_masks"].sum() == 10  # half the synthetic data is IR007
