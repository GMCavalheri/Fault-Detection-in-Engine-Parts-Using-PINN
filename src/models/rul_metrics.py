"""Standard C-MAPSS RUL evaluation metrics."""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The PHM08/C-MAPSS scoring function (Saxena & Goebel, 2008).

    Asymmetric around zero error: predicting a HIGHER RUL than the truth
    (`d >= 0`, overestimating remaining life) is penalized far more heavily
    than predicting a lower one (`d < 0`), since overestimating how much
    life is left before failure is the operationally dangerous direction.
    Lower is better; unlike RMSE this isn't in cycle units, only useful for
    relative comparison between models on the same test set.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    early = d[d < 0]
    late = d[d >= 0]
    return float(np.sum(np.exp(-early / 13) - 1) + np.sum(np.exp(late / 10) - 1))
