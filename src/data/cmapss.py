"""Loading, RUL labeling, and sequence windowing for NASA C-MAPSS.

Standard practices from the RUL-regression literature (e.g. Saxena & Goebel's
original PHM08 challenge, Zheng et al. 2017's LSTM baseline), applied here so
results are comparable to published work on this exact benchmark:

- **Piecewise-linear RUL capping**: raw RUL (cycles until failure) is only
  meaningful close to failure - early in an engine's life nothing has
  degraded yet, so a linear RUL target (e.g. 350 cycles out) is unlearnable
  noise. Training labels are capped at `RUL_CAP` cycles; evaluation compares
  against the dataset's own uncapped test RUL values (the field's standard
  convention, see docs/cmapss_results.md).
- **Sliding-window sequences**: a fixed-length window of the last
  `window_size` cycles' sensor readings predicts the RUL at the window's
  last cycle. Short trajectories are front-padded by repeating the first
  row (standard practice, avoids a variable-length special case).
- **Sensor selection by training-set variance**: FD001's single operating
  condition leaves several sensors and all three operational settings
  constant (or near it) - selected dynamically from train data, not
  hardcoded, so this still works if FD002-4 (six conditions) are added later.
"""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SENSOR_COLS = [c for c in COLUMNS if c.startswith("sensor_")]
OP_SETTING_COLS = [c for c in COLUMNS if c.startswith("op_setting_")]

RUL_CAP = 125
DEFAULT_WINDOW_SIZE = 30


def load_subset(cmapss_dir: Path, subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Returns (train_df, test_df, test_true_rul). `test_true_rul[i]` is the
    true RUL for the unit at the i-th position of `RUL_{subset}.txt`, which
    corresponds to `sorted(test_df["unit_id"].unique())[i]` (the dataset's
    documented convention).
    """
    train_df = pd.read_csv(cmapss_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test_df = pd.read_csv(cmapss_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test_true_rul = pd.read_csv(cmapss_dir / f"RUL_{subset}.txt", header=None).iloc[:, 0].to_numpy(dtype=np.float64)
    return train_df, test_df, test_true_rul


def add_capped_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df


def select_informative_columns(
    train_df: pd.DataFrame, candidate_cols: list[str] = SENSOR_COLS + OP_SETTING_COLS, std_threshold: float = 1e-5
) -> list[str]:
    """Drops columns that are ~constant in the training data - computed from
    train only, so this never leaks test-set information into feature choice.
    """
    stds = train_df[candidate_cols].std()
    return [c for c in candidate_cols if stds[c] > std_threshold]


def _windows_for_unit(values: np.ndarray, window_size: int) -> np.ndarray:
    n = len(values)
    if n < window_size:
        pad = np.repeat(values[:1], window_size - n, axis=0)
        values = np.vstack([pad, values])
    return values


def build_train_sequences(
    df: pd.DataFrame, feature_cols: list[str], window_size: int = DEFAULT_WINDOW_SIZE, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One sequence per (unit, cycle >= window_size), label = that cycle's
    capped RUL. Returns (sequences[N, window_size, n_features], targets[N], unit_ids[N]).
    """
    sequences, targets, unit_ids = [], [], []
    for unit_id, group in df.groupby("unit_id"):
        group = group.sort_values("cycle")
        values = _windows_for_unit(group[feature_cols].to_numpy(dtype=np.float64), window_size)
        ruls = group["RUL"].to_numpy(dtype=np.float64)
        if len(ruls) < window_size:
            ruls = np.concatenate([np.full(window_size - len(ruls), ruls[0]), ruls])
        n = len(values)
        for end in range(window_size, n + 1, stride):
            sequences.append(values[end - window_size : end])
            targets.append(ruls[end - 1])
            unit_ids.append(unit_id)
    return np.array(sequences), np.array(targets, dtype=np.float64), np.array(unit_ids)


def build_test_sequences(
    df: pd.DataFrame, feature_cols: list[str], window_size: int = DEFAULT_WINDOW_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """One sequence per unit: its last `window_size` cycles (the dataset's
    own truncated-trajectory test protocol). Returns (sequences, unit_ids)
    with unit_ids sorted ascending, matching RUL_{subset}.txt's row order.
    """
    sequences, unit_ids = [], []
    for unit_id, group in df.sort_values("unit_id").groupby("unit_id", sort=True):
        group = group.sort_values("cycle")
        values = _windows_for_unit(group[feature_cols].to_numpy(dtype=np.float64), window_size)
        sequences.append(values[-window_size:])
        unit_ids.append(unit_id)
    return np.array(sequences), np.array(unit_ids)
