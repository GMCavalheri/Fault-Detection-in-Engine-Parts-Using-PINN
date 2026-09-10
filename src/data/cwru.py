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


def load_de_channel(path: Path) -> np.ndarray:
    """Load the drive-end accelerometer channel, the standard CWRU channel."""
    mat = loadmat(path)
    for key in mat:
        if key.endswith("_DE_time"):
            return mat[key].ravel().astype(np.float64)
    raise KeyError(f"No *_DE_time channel found in {path.name}")


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
