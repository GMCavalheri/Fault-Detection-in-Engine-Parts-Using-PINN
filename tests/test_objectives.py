import numpy as np
from torch.utils.data import DataLoader

from src.data.torch_utils import WindowDataset, compute_train_stats
from src.optimization.objectives import make_cnn_objective, make_svm_objective

WINDOW_SIZE = 2048


def test_svm_objective_returns_valid_accuracy():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 5))
    y_train = rng.integers(0, 2, size=30)
    X_val = rng.normal(size=(10, 5))
    y_val = rng.integers(0, 2, size=10)

    objective = make_svm_objective(X_train, y_train, X_val, y_val)
    score = objective({"C": 1.0, "gamma": 0.1})
    assert 0.0 <= score <= 1.0


def test_cnn_objective_returns_valid_accuracy():
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(16, WINDOW_SIZE))
    targets = rng.integers(0, 3, size=16)
    mean, std = compute_train_stats(windows)
    ds = WindowDataset(windows, targets, mean, std)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    objective = make_cnn_objective(loader, loader, num_classes=3, epochs=1)
    score = objective({"lr": 1e-3, "dropout": 0.1, "width_mult": 1.0})
    assert 0.0 <= score <= 1.0
