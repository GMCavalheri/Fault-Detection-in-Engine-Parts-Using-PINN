"""Parses an uploaded vibration signal: a CWRU-style .mat file (any variable
ending in `_DE_time`) or a plain .csv/.txt file of one float per line/column.
"""

import io
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def parse_signal_upload(filename: str, data: bytes) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    if suffix == ".mat":
        return _parse_mat(data, filename)
    if suffix in (".csv", ".txt"):
        return _parse_text(data)
    raise ValueError(f"Unsupported file type '{suffix}' - upload a .mat, .csv, or .txt file")


def _parse_mat(data: bytes, filename: str) -> np.ndarray:
    mat = loadmat(io.BytesIO(data))
    candidates = [k for k in mat if k.endswith("_DE_time")]
    if not candidates:
        raise ValueError(f"No *_DE_time channel found in {filename}")
    # if multiple variables match (e.g. a stray leftover from another export,
    # see docs/datasets.md), take the largest as the file's own real signal
    key = max(candidates, key=lambda k: mat[k].size)
    return mat[key].ravel().astype(np.float64)


def _parse_text(data: bytes) -> np.ndarray:
    values = np.loadtxt(io.StringIO(data.decode("utf-8")))
    return np.asarray(values, dtype=np.float64).ravel()
