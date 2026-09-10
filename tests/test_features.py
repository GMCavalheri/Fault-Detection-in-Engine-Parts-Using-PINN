import numpy as np
import pytest

from src.data.features import extract_features, time_domain_features

FS = 12_000


def _sine_window(freq_hz: float, amplitude: float, n_samples: int = 2048) -> np.ndarray:
    t = np.arange(n_samples) / FS
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def test_time_domain_features_on_sine_wave():
    amplitude = 2.0
    window = _sine_window(freq_hz=500, amplitude=amplitude)
    feats = time_domain_features(window)

    assert feats["td_rms"] == pytest.approx(amplitude / np.sqrt(2), rel=1e-3)
    assert feats["td_peak"] == pytest.approx(amplitude, rel=1e-2)
    assert feats["td_crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-2)
    # window doesn't span an integer number of periods, so mean is small but not exactly 0
    assert feats["td_mean"] == pytest.approx(0.0, abs=0.05)


def test_frequency_domain_finds_dominant_frequency():
    window = _sine_window(freq_hz=1200, amplitude=1.0)
    feats = extract_features(window, fs=FS)

    # FFT bin resolution at 2048 samples / 12kHz is ~5.9 Hz
    assert feats["fd_dominant_freq_hz"] == pytest.approx(1200, abs=10)


def test_extract_features_returns_finite_values():
    rng = np.random.default_rng(0)
    window = rng.normal(size=2048)
    feats = extract_features(window, fs=FS)

    assert all(np.isfinite(v) for v in feats.values())
    assert len(feats) == 10 + 5 + 6  # time + frequency + envelope feature counts
