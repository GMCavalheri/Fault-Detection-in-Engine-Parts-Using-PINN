import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cmapss_torch import RulSequenceDataset, compute_sequence_train_stats
from src.models.lstm_rul import LstmRul, predict, train_lstm
from src.models.rul_baseline import summarize_windows, train_gbr

WINDOW_SIZE = 10
N_FEATURES = 4


def _synthetic_sequences(n_samples: int):
    rng = np.random.default_rng(0)
    sequences = rng.normal(size=(n_samples, WINDOW_SIZE, N_FEATURES))
    targets = rng.uniform(0, 125, size=n_samples)
    return sequences, targets


def test_summarize_windows_shape_and_values():
    sequences = np.arange(2 * 3 * 2).reshape(2, 3, 2).astype(float)
    summary = summarize_windows(sequences)

    assert summary.shape == (2, 2 * 3)  # n_features * 3 (mean, std, last)
    np.testing.assert_allclose(summary[0, :2], sequences[0].mean(axis=0))
    np.testing.assert_allclose(summary[0, 4:], sequences[0, -1])


def test_train_gbr_fits_and_predicts_finite_values():
    sequences, targets = _synthetic_sequences(50)
    X = summarize_windows(sequences)
    model = train_gbr(X, targets)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_compute_sequence_train_stats_shapes():
    sequences, _ = _synthetic_sequences(20)
    mean, std = compute_sequence_train_stats(sequences)
    assert mean.shape == (N_FEATURES,)
    assert std.shape == (N_FEATURES,)
    assert np.all(std > 0)


def test_lstm_rul_forward_pass_shape():
    model = LstmRul(n_features=N_FEATURES, hidden_size=8, num_layers=1)
    x = torch.randn(5, WINDOW_SIZE, N_FEATURES)
    out = model(x)
    assert out.shape == (5,)


def test_train_lstm_smoke_test():
    sequences, targets = _synthetic_sequences(20)
    mean, std = compute_sequence_train_stats(sequences)
    ds = RulSequenceDataset(sequences, targets, mean, std)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    model = LstmRul(n_features=N_FEATURES, hidden_size=8, num_layers=1)
    history = train_lstm(model, loader, loader, epochs=1)

    assert len(history) == 1
    assert set(history[0]) == {"epoch", "train_rmse", "val_rmse"}
    assert history[0]["train_rmse"] >= 0

    preds, actual_targets = predict(model, loader)
    assert preds.shape == actual_targets.shape == (20,)
