import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.torch_utils import WindowDataset, compute_train_stats
from src.models.autoencoder import ConvAutoencoder, reconstruction_errors, train_autoencoder
from src.models.cnn import Cnn1D, predict, train_cnn

WINDOW_SIZE = 2048


def _synthetic_dataset(n_samples: int, n_classes: int) -> WindowDataset:
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(n_samples, WINDOW_SIZE))
    targets = rng.integers(0, n_classes, size=n_samples)
    mean, std = compute_train_stats(windows)
    return WindowDataset(windows, targets, mean, std)


def test_cnn_forward_pass_shape():
    model = Cnn1D(num_classes=10)
    x = torch.randn(4, 1, WINDOW_SIZE)
    assert model(x).shape == (4, 10)


def test_autoencoder_reconstructs_input_shape():
    model = ConvAutoencoder()
    x = torch.randn(4, 1, WINDOW_SIZE)
    assert model(x).shape == x.shape


def test_train_cnn_one_epoch_smoke_test():
    ds = _synthetic_dataset(n_samples=20, n_classes=3)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = Cnn1D(num_classes=3)

    history = train_cnn(model, loader, loader, epochs=1)
    assert len(history) == 1
    assert set(history[0]) == {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}

    preds, targets = predict(model, loader)
    assert preds.shape == targets.shape == (20,)


def test_train_autoencoder_and_reconstruction_errors():
    ds = _synthetic_dataset(n_samples=16, n_classes=1)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = ConvAutoencoder()

    history = train_autoencoder(model, loader, epochs=1)
    assert len(history) == 1 and "train_loss" in history[0]

    errors = reconstruction_errors(model, loader)
    assert errors.shape == (16,)
    assert np.all(errors >= 0)
