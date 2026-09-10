"""Loading and windowing for the CWRU bearing dataset.

File naming convention used in data/raw/cwru/ (see docs/datasets.md):
  Normal_{load}.mat            - baseline, no fault
  IR{diameter}_{load}.mat      - inner race fault
  B{diameter}_{load}.mat       - ball fault
  OR{diameter}at6_{load}.mat   - outer race fault, centered at 6:00

`load` is motor load in HP (0-3). `diameter` is fault diameter in thousandths
of an inch (007/014/021). Each .mat file stores its channels under
variable names with a numeric file-ID prefix, e.g. `X097_DE_time`, so keys
are located by suffix rather than assumed to be fixed.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SAMPLE_RATE_HZ = 12_000

_FILENAME_RE = re.compile(
    r"^(?P<fault_type>Normal|IR|B|OR)(?P<diameter>\d{3})?(?:at6)?_(?P<load>\d)$"
)


@dataclass(frozen=True)
class CwruFile:
    path: Path
    fault_type: str  # "Normal", "IR", "B", "OR"
    diameter: str  # "" for Normal, else "007"/"014"/"021"
    load: int  # motor load in HP, 0-3

    @property
    def label(self) -> str:
        return self.fault_type if self.fault_type == "Normal" else f"{self.fault_type}{self.diameter}"


def parse_filename(path: Path) -> CwruFile:
    match = _FILENAME_RE.match(path.stem)
    if not match:
        raise ValueError(f"Unrecognized CWRU filename: {path.name}")
    groups = match.groupdict()
    return CwruFile(
        path=path,
        fault_type=groups["fault_type"],
        diameter=groups["diameter"] or "",
        load=int(groups["load"]),
    )


def list_cwru_files(cwru_dir: Path) -> list[CwruFile]:
    return sorted(
        (parse_filename(p) for p in cwru_dir.glob("*.mat")),
        key=lambda f: (f.fault_type, f.diameter, f.load),
    )


_KEY_PREFIX_RE = re.compile(r"^X(\d+)")


def _select_key(mat: dict, suffix: str, path: Path) -> str:
    candidates = [k for k in mat if k.endswith(suffix)]
    if not candidates:
        raise KeyError(f"No *{suffix} channel found in {path.name}")
    if len(candidates) == 1:
        return candidates[0]
    # A couple of CWRU's own files (e.g. Normal_2.mat) carry stray leftover
    # variables from an earlier export alongside their own - observed as an
    # older (numerically smaller) X-prefix left in the workspace when the
    # file was prepared. The file's own data has the highest numeric prefix.
    return max(candidates, key=lambda k: int(_KEY_PREFIX_RE.match(k).group(1)))


def load_channel(path: Path, suffix: str) -> np.ndarray:
    """Load one accelerometer channel by its variable-name suffix, e.g. '_DE_time'."""
    mat = loadmat(path)
    key = _select_key(mat, suffix, path)
    return mat[key].ravel().astype(np.float64)


def load_de_channel(path: Path) -> np.ndarray:
    """Load the drive-end accelerometer channel, the standard CWRU channel."""
    return load_channel(path, "_DE_time")


def load_rpm(path: Path) -> float:
    """Load the recorded shaft speed (constant scalar per file, e.g. `X097RPM`)."""
    mat = loadmat(path)
    key = _select_key(mat, "RPM", path)
    return float(mat[key].ravel()[0])


def window_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Chop a 1D signal into non-overlapping windows, dropping the remainder.

    Non-overlapping windows never share samples, so windows from one file
    don't leak into each other directly - but see docs/datasets.md: windows
    from the SAME file are still correlated (same recording session) and
    must not be split across train/test.
    """
    n_windows = len(signal) // window_size
    trimmed = signal[: n_windows * window_size]
    return trimmed.reshape(n_windows, window_size)


def build_windowed_dataset(
    cwru_dir: Path, window_size: int = 2048
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (windows, labels, loads, file_ids) arrays across all CWRU files.

    `file_ids` gives each source file a stable integer id, so callers can
    split by file (or by time-order within a file) without leaking windows
    from one recording across train and test.
    """
    files = list_cwru_files(cwru_dir)

    all_windows = []
    all_labels = []
    all_loads = []
    all_file_ids = []
    for file_id, f in enumerate(files):
        signal = load_de_channel(f.path)
        windows = window_signal(signal, window_size)
        all_windows.append(windows)
        all_labels.extend([f.label] * len(windows))
        all_loads.extend([f.load] * len(windows))
        all_file_ids.extend([file_id] * len(windows))

    return (
        np.concatenate(all_windows, axis=0),
        np.array(all_labels),
        np.array(all_loads),
        np.array(all_file_ids),
    )


# A couple of CWRU's own published files (Normal_1.mat, Normal_2.mat) are missing the RPM
# field entirely - a known quirk of this dataset. Fall back to the nominal per-load RPM
# published by CWRU and used throughout the literature (matches every other file's measured
# RPM at the same load to within ~1-2 rpm, see docs/datasets.md).
NOMINAL_RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def build_windowed_dataset_with_rpm(
    cwru_dir: Path, window_size: int = 2048
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like build_windowed_dataset, plus each window's recorded shaft RPM
    (constant within a file, broadcast to every window from it) - needed for
    Phase 5's physics-informed loss, where the expected bearing defect
    frequency is a function of shaft speed (src/physics/bearing.py).

    Kept as a separate function (rather than changing build_windowed_dataset's
    return signature) so every notebook from Phases 1-4 stays reproducible.
    """
    files = list_cwru_files(cwru_dir)

    all_windows, all_labels, all_loads, all_file_ids, all_rpms = [], [], [], [], []
    for file_id, f in enumerate(files):
        signal = load_de_channel(f.path)
        windows = window_signal(signal, window_size)
        try:
            rpm = load_rpm(f.path)
        except KeyError:
            rpm = NOMINAL_RPM_BY_LOAD[f.load]
        all_windows.append(windows)
        all_labels.extend([f.label] * len(windows))
        all_loads.extend([f.load] * len(windows))
        all_file_ids.extend([file_id] * len(windows))
        all_rpms.extend([rpm] * len(windows))

    return (
        np.concatenate(all_windows, axis=0),
        np.array(all_labels),
        np.array(all_loads),
        np.array(all_file_ids),
        np.array(all_rpms, dtype=np.float64),
    )


CHANNEL_SUFFIXES = {"DE": "_DE_time", "FE": "_FE_time", "BA": "_BA_time"}


def build_multichannel_windowed_dataset(
    cwru_dir: Path, window_size: int = 2048, channels: tuple[str, ...] = ("DE", "FE")
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Like build_windowed_dataset, but keeps multiple channels aligned per window.

    Returns (per_channel_windows, labels, loads, file_ids) where
    per_channel_windows[channel] has the same shape and window-for-window
    alignment across channels - so windows[c][i] for different c are all the
    same time segment of the same file. Requested channels must be present
    in every file (DE and FE both are; BA is not, see docs/datasets.md).
    """
    files = list_cwru_files(cwru_dir)

    per_channel_windows: dict[str, list[np.ndarray]] = {c: [] for c in channels}
    all_labels, all_loads, all_file_ids = [], [], []
    for file_id, f in enumerate(files):
        channel_signals = {c: load_channel(f.path, CHANNEL_SUFFIXES[c]) for c in channels}
        # channels are recorded simultaneously but occasionally differ by a few samples
        n_windows = min(len(sig) // window_size for sig in channel_signals.values())

        for c, sig in channel_signals.items():
            per_channel_windows[c].append(window_signal(sig[: n_windows * window_size], window_size))

        all_labels.extend([f.label] * n_windows)
        all_loads.extend([f.load] * n_windows)
        all_file_ids.extend([file_id] * n_windows)

    return (
        {c: np.concatenate(ws, axis=0) for c, ws in per_channel_windows.items()},
        np.array(all_labels),
        np.array(all_loads),
        np.array(all_file_ids),
    )
