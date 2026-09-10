from pathlib import Path

import numpy as np

from src.data.cwru import parse_filename, window_signal


def test_parse_filename_normal():
    f = parse_filename(Path("Normal_2.mat"))
    assert f.fault_type == "Normal"
    assert f.diameter == ""
    assert f.load == 2
    assert f.label == "Normal"


def test_parse_filename_inner_race():
    f = parse_filename(Path("IR007_1.mat"))
    assert f.fault_type == "IR"
    assert f.diameter == "007"
    assert f.load == 1
    assert f.label == "IR007"


def test_parse_filename_outer_race():
    f = parse_filename(Path("OR021at6_0.mat"))
    assert f.fault_type == "OR"
    assert f.diameter == "021"
    assert f.load == 0
    assert f.label == "OR021"


def test_window_signal_drops_remainder_and_does_not_overlap():
    signal = np.arange(5000)
    windows = window_signal(signal, window_size=2048)

    assert windows.shape == (2, 2048)
    assert np.array_equal(windows[0], signal[:2048])
    assert np.array_equal(windows[1], signal[2048:4096])
