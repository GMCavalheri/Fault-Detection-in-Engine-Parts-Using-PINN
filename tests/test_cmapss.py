import numpy as np
import pandas as pd

from src.data.cmapss import (
    add_capped_rul,
    build_test_sequences,
    build_train_sequences,
    fit_regime_normalizer,
    select_informative_columns,
)


def _make_unit_df(unit_id: int, n_cycles: int, sensor_value_fn=lambda c: float(c)) -> pd.DataFrame:
    cycles = np.arange(1, n_cycles + 1)
    return pd.DataFrame(
        {
            "unit_id": unit_id,
            "cycle": cycles,
            "sensor_a": [sensor_value_fn(c) for c in cycles],
            "sensor_b": 42.0,  # constant
        }
    )


def test_add_capped_rul_counts_down_to_failure_and_caps():
    df = _make_unit_df(1, n_cycles=200)
    result = add_capped_rul(df, cap=125)

    # cycle 1 of a 200-cycle run: raw RUL = 199, capped to 125
    assert result.loc[result["cycle"] == 1, "RUL"].item() == 125
    # cycle 200 (last cycle): RUL = 0
    assert result.loc[result["cycle"] == 200, "RUL"].item() == 0
    # cycle 100: raw RUL = 100, still under the cap
    assert result.loc[result["cycle"] == 100, "RUL"].item() == 100


def test_select_informative_columns_drops_constant_sensor():
    df = _make_unit_df(1, n_cycles=50)
    selected = select_informative_columns(df, candidate_cols=["sensor_a", "sensor_b"])
    assert selected == ["sensor_a"]


def test_build_train_sequences_shapes_and_labels():
    df = add_capped_rul(_make_unit_df(1, n_cycles=50), cap=125)
    sequences, targets, unit_ids = build_train_sequences(df, feature_cols=["sensor_a"], window_size=10, stride=1)

    # windows end at cycles 10..50 -> 41 windows
    assert sequences.shape == (41, 10, 1)
    assert len(targets) == len(unit_ids) == 41
    assert set(unit_ids) == {1}
    # first window ends at cycle 10 -> raw RUL = 50-10 = 40
    assert targets[0] == 40
    # last window ends at cycle 50 (last cycle) -> RUL = 0
    assert targets[-1] == 0
    # the window itself should be sensor values for cycles 1..10
    np.testing.assert_allclose(sequences[0].ravel(), np.arange(1, 11))


def test_build_train_sequences_pads_short_trajectory():
    df = add_capped_rul(_make_unit_df(1, n_cycles=5), cap=125)
    sequences, targets, unit_ids = build_train_sequences(df, feature_cols=["sensor_a"], window_size=10, stride=1)

    # trajectory shorter than window_size -> exactly one (padded) window
    assert sequences.shape == (1, 10, 1)
    # front-padded by repeating cycle 1's value
    np.testing.assert_allclose(sequences[0, :5].ravel(), 1.0)
    np.testing.assert_allclose(sequences[0, 5:].ravel(), np.arange(1, 6))


def test_build_test_sequences_takes_last_window_and_sorts_by_unit():
    df = pd.concat(
        [_make_unit_df(2, n_cycles=50), _make_unit_df(1, n_cycles=30)], ignore_index=True
    )
    sequences, unit_ids = build_test_sequences(df, feature_cols=["sensor_a"], window_size=10)

    assert list(unit_ids) == [1, 2]  # sorted ascending, matching RUL_FD00X.txt row order
    assert sequences.shape == (2, 10, 1)
    # unit 1's window: last 10 cycles of a 30-cycle run -> values 21..30
    np.testing.assert_allclose(sequences[0].ravel(), np.arange(21, 31))
    # unit 2's window: last 10 cycles of a 50-cycle run -> values 41..50
    np.testing.assert_allclose(sequences[1].ravel(), np.arange(41, 51))


def test_build_test_sequences_pads_short_trajectory():
    df = _make_unit_df(1, n_cycles=4)
    sequences, unit_ids = build_test_sequences(df, feature_cols=["sensor_a"], window_size=10)

    assert sequences.shape == (1, 10, 1)
    np.testing.assert_allclose(sequences[0, :6].ravel(), 1.0)
    np.testing.assert_allclose(sequences[0, 6:].ravel(), np.arange(1, 5))


def _make_two_regime_df(n_per_regime: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # regime A: op_setting_1=0 -> sensor offset 500; regime B: op_setting_1=100 -> sensor offset 600
    op_setting_1 = np.concatenate([np.zeros(n_per_regime), np.full(n_per_regime, 100.0)])
    offsets = np.concatenate([np.full(n_per_regime, 500.0), np.full(n_per_regime, 600.0)])
    sensor = offsets + rng.normal(scale=1.0, size=2 * n_per_regime)  # small degradation-independent noise
    return pd.DataFrame({"op_setting_1": op_setting_1, "sensor_x": sensor})


def test_regime_normalizer_removes_condition_driven_offset():
    df = _make_two_regime_df()
    normalizer = fit_regime_normalizer(df, sensor_cols=["sensor_x"], op_setting_cols=["op_setting_1"], n_regimes=2)
    normalized = normalizer.transform(df)

    # before normalization the two regimes have very different means (500 vs 600)
    raw_means = df.groupby("op_setting_1")["sensor_x"].mean()
    assert abs(raw_means.iloc[0] - raw_means.iloc[1]) > 50

    # after normalization both regimes should be centered near 0 with unit-ish scale
    normalized_means = normalized.groupby(df["op_setting_1"])["sensor_x"].mean()
    assert (normalized_means.abs() < 0.3).all()
    normalized_stds = normalized.groupby(df["op_setting_1"])["sensor_x"].std()
    assert ((normalized_stds - 1.0).abs() < 0.3).all()


def test_regime_normalizer_transform_does_not_refit_on_new_data():
    train_df = _make_two_regime_df(n_per_regime=200)
    normalizer = fit_regime_normalizer(
        train_df, sensor_cols=["sensor_x"], op_setting_cols=["op_setting_1"], n_regimes=2
    )

    # a "test" set with only regime A present - must still normalize using train-fit regime A stats,
    # not re-derive statistics from this smaller/different sample
    test_df = _make_two_regime_df(n_per_regime=5)
    test_df = test_df[test_df["op_setting_1"] == 0.0].reset_index(drop=True)
    normalized_test = normalizer.transform(test_df)

    assert abs(normalized_test["sensor_x"].mean()) < 1.0  # centered near 0, not wildly off
