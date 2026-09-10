import numpy as np
import pytest
from scipy.io import savemat

from src.data.cwru import build_windowed_dataset_with_rpm, load_rpm


def _write_fake_cwru_file(path, prefix: str, rpm: float, n_samples: int = 4096):
    savemat(
        path,
        {
            f"X{prefix}_DE_time": np.random.default_rng(0).normal(size=(n_samples, 1)),
            f"X{prefix}RPM": np.array([[rpm]], dtype=np.uint16),
        },
    )


def test_load_rpm_reads_scalar_from_mat_file(tmp_path):
    file_path = tmp_path / "IR007_0.mat"
    _write_fake_cwru_file(file_path, prefix="105", rpm=1797)

    assert load_rpm(file_path) == pytest.approx(1797)


def test_build_windowed_dataset_with_rpm_broadcasts_rpm_per_file(tmp_path):
    _write_fake_cwru_file(tmp_path / "Normal_0.mat", prefix="097", rpm=1797, n_samples=4096)
    _write_fake_cwru_file(tmp_path / "IR007_1.mat", prefix="106", rpm=1772, n_samples=4096)

    windows, labels, loads, file_ids, rpms = build_windowed_dataset_with_rpm(tmp_path, window_size=2048)

    assert len(windows) == len(labels) == len(loads) == len(file_ids) == len(rpms)
    # 2 windows per file (4096 / 2048), 2 files
    assert len(windows) == 4
    assert set(rpms[labels == "Normal"]) == {1797.0}
    assert set(rpms[labels == "IR007"]) == {1772.0}
