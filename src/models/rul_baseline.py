"""Classical ML baseline for RUL regression: summarizes each sliding window
into flat statistics, then a Gradient Boosting regressor - the same
"classical vs. deep learning" comparison pattern as Phases 1-2's fault
classification, applied to RUL.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def summarize_windows(sequences: np.ndarray) -> np.ndarray:
    """(N, window_size, n_features) -> (N, n_features*3): per-feature mean,
    std, and last-cycle value across the window - a classical-ML-friendly
    summary of the same sequence the LSTM consumes directly.
    """
    mean = sequences.mean(axis=1)
    std = sequences.std(axis=1)
    last = sequences[:, -1, :]
    return np.concatenate([mean, std, last], axis=1)


def train_gbr(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=random_state)
    model.fit(X_train, y_train)
    return model
