"""Feature extraction for vibration signal windows.

Three families of hand-engineered features, standard in vibration-based
fault diagnosis literature:
  - time-domain statistics (shape of the raw waveform)
  - frequency-domain statistics (shape of the FFT magnitude spectrum)
  - envelope-spectrum statistics (Hilbert-transform amplitude envelope,
    which surfaces the periodic impacts characteristic of bearing faults)
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

_EPS = 1e-12


def time_domain_features(window: np.ndarray) -> dict[str, float]:
    abs_window = np.abs(window)
    rms = np.sqrt(np.mean(window**2))
    peak = np.max(abs_window)
    mean_abs = np.mean(abs_window)
    return {
        "td_mean": float(np.mean(window)),
        "td_std": float(np.std(window)),
        "td_rms": float(rms),
        "td_peak": float(peak),
        "td_peak_to_peak": float(np.max(window) - np.min(window)),
        "td_skew": float(skew(window)),
        "td_kurtosis": float(kurtosis(window)),
        "td_crest_factor": float(peak / (rms + _EPS)),
        "td_shape_factor": float(rms / (mean_abs + _EPS)),
        "td_impulse_factor": float(peak / (mean_abs + _EPS)),
    }


def frequency_domain_features(window: np.ndarray, fs: float) -> dict[str, float]:
    spectrum = np.abs(rfft(window))
    freqs = rfftfreq(len(window), d=1 / fs)
    total_energy = np.sum(spectrum**2) + _EPS
    centroid = float(np.sum(freqs * spectrum) / (np.sum(spectrum) + _EPS))
    dominant_freq = float(freqs[np.argmax(spectrum)])
    return {
        "fd_centroid_hz": centroid,
        "fd_dominant_freq_hz": dominant_freq,
        "fd_spectral_std": float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum**2) / total_energy)),
        "fd_energy": float(np.sum(spectrum**2)),
        "fd_mean_magnitude": float(np.mean(spectrum)),
    }


def envelope_features(window: np.ndarray, fs: float) -> dict[str, float]:
    envelope = np.abs(hilbert(window))
    env_spectrum = np.abs(rfft(envelope - np.mean(envelope)))
    freqs = rfftfreq(len(envelope), d=1 / fs)
    return {
        "env_mean": float(np.mean(envelope)),
        "env_std": float(np.std(envelope)),
        "env_peak": float(np.max(envelope)),
        "env_kurtosis": float(kurtosis(envelope)),
        "env_spectrum_peak_freq_hz": float(freqs[np.argmax(env_spectrum)]),
        "env_spectrum_energy": float(np.sum(env_spectrum**2)),
    }


def extract_features(window: np.ndarray, fs: float) -> dict[str, float]:
    return {
        **time_domain_features(window),
        **frequency_domain_features(window, fs),
        **envelope_features(window, fs),
    }


def build_feature_matrix(windows: np.ndarray, fs: float) -> pd.DataFrame:
    rows = [extract_features(w, fs) for w in windows]
    return pd.DataFrame(rows)
