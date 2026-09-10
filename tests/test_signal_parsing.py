import io

import numpy as np
import pytest
from scipy.io import savemat

from src.api.signal_parsing import parse_signal_upload


def test_parse_csv_upload():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    buf = io.StringIO()
    np.savetxt(buf, values)
    data = parse_signal_upload("signal.csv", buf.getvalue().encode("utf-8"))
    np.testing.assert_allclose(data, values)


def test_parse_txt_upload():
    values = np.array([5.0, 6.0, 7.0])
    buf = io.StringIO()
    np.savetxt(buf, values)
    data = parse_signal_upload("signal.txt", buf.getvalue().encode("utf-8"))
    np.testing.assert_allclose(data, values)


def test_parse_mat_upload_finds_de_time_channel():
    buf = io.BytesIO()
    values = np.random.default_rng(0).normal(size=(100, 1))
    savemat(buf, {"X999_DE_time": values, "X999RPM": np.array([[1797]])})
    data = parse_signal_upload("signal.mat", buf.getvalue())
    np.testing.assert_allclose(data, values.ravel())


def test_parse_mat_upload_picks_largest_when_multiple_de_time_keys():
    buf = io.BytesIO()
    small = np.random.default_rng(0).normal(size=(10, 1))
    large = np.random.default_rng(1).normal(size=(50, 1))
    savemat(buf, {"X098_DE_time": small, "X099_DE_time": large})
    data = parse_signal_upload("signal.mat", buf.getvalue())
    assert len(data) == 50
    np.testing.assert_allclose(data, large.ravel())


def test_parse_mat_upload_without_de_time_raises():
    buf = io.BytesIO()
    savemat(buf, {"something_else": np.array([[1, 2, 3]])})
    with pytest.raises(ValueError, match="DE_time"):
        parse_signal_upload("signal.mat", buf.getvalue())


def test_parse_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_signal_upload("signal.xyz", b"whatever")
