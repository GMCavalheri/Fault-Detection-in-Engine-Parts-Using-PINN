import numpy as np
import pytest

from src.models.rul_metrics import nasa_score, rmse


def test_rmse_zero_for_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx((3**2 + 4**2) ** 0.5 / (2**0.5))


def test_nasa_score_zero_for_perfect_predictions():
    y = np.array([50.0, 100.0])
    assert nasa_score(y, y) == pytest.approx(0.0)


def test_nasa_score_penalizes_late_predictions_more_than_early_of_same_magnitude():
    y_true = np.array([50.0])
    late = nasa_score(y_true, y_true + 10)  # overestimate RUL by 10 - the dangerous direction
    early = nasa_score(y_true, y_true - 10)  # underestimate RUL by 10 - the conservative direction
    assert late > early


def test_nasa_score_grows_with_error_magnitude():
    y_true = np.array([50.0])
    small_late = nasa_score(y_true, y_true + 5)
    large_late = nasa_score(y_true, y_true + 20)
    assert large_late > small_late
