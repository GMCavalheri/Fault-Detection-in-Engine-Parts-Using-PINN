"""Rolling-element bearing defect frequencies - the governing kinematic
relationship for Phase 5's physics-informed loss.

A rolling-element bearing generates a periodic impact at a frequency set by
its geometry and shaft speed whenever a defect on a given component (outer
race, inner race, or a ball) passes through the load zone. For a bearing
with `n_balls` balls of diameter `d` on a pitch circle of diameter `D`,
contact angle `phi`, and shaft rotating at `f_r` Hz:

    BPFO = (n/2) * f_r * (1 - (d/D) cos(phi))   - outer race defect frequency
    BPFI = (n/2) * f_r * (1 + (d/D) cos(phi))   - inner race defect frequency
    BSF  = (D/(2d)) * f_r * (1 - (d/D)^2 cos(phi)^2)  - ball (rolling element) defect frequency
    FTF  = (f_r/2) * (1 - (d/D) cos(phi))       - cage/train frequency

This is the standard formulation used throughout the bearing-diagnostics
literature (e.g. Randall & Antoni, "Rolling element bearing diagnostics - A
tutorial", Mechanical Systems and Signal Processing, 2011).

CWRU seeds every fault in this project's dataset on the **drive-end**
bearing, an SKF 6205-2RS JEM deep-groove ball bearing (9 balls, contact
angle ~0 deg) - the geometry below reproduces the defect-frequency
multipliers (in orders of shaft speed) published by the CWRU Bearing Data
Center: BPFO ~3.585x, BPFI ~5.415x, BSF ~2.357x, FTF ~0.3983x.
"""

import numpy as np

# SKF 6205-2RS JEM (CWRU drive-end bearing) geometry, inches
N_BALLS = 9
BALL_DIAMETER_IN = 0.3126
PITCH_DIAMETER_IN = 1.537
CONTACT_ANGLE_DEG = 0.0

_RATIO = BALL_DIAMETER_IN / PITCH_DIAMETER_IN
_COS_PHI = np.cos(np.deg2rad(CONTACT_ANGLE_DEG))


def shaft_frequency_hz(rpm: float) -> float:
    return rpm / 60.0


def bpfo_hz(rpm: float) -> float:
    """Ball Pass Frequency, Outer race - defect frequency for an OR fault."""
    return (N_BALLS / 2) * shaft_frequency_hz(rpm) * (1 - _RATIO * _COS_PHI)


def bpfi_hz(rpm: float) -> float:
    """Ball Pass Frequency, Inner race - defect frequency for an IR fault."""
    return (N_BALLS / 2) * shaft_frequency_hz(rpm) * (1 + _RATIO * _COS_PHI)


def bsf_hz(rpm: float) -> float:
    """Ball Spin Frequency - defect frequency for a Ball (B) fault."""
    return (PITCH_DIAMETER_IN / (2 * BALL_DIAMETER_IN)) * shaft_frequency_hz(rpm) * (
        1 - (_RATIO * _COS_PHI) ** 2
    )


def ftf_hz(rpm: float) -> float:
    """Fundamental Train Frequency (cage speed) - not fault-specific, included for reference."""
    return 0.5 * shaft_frequency_hz(rpm) * (1 - _RATIO * _COS_PHI)


_FAULT_TYPE_TO_FREQUENCY_FN = {"IR": bpfi_hz, "B": bsf_hz, "OR": bpfo_hz}


def expected_fault_frequency_hz(label: str, rpm: float) -> float | None:
    """Physics-predicted defect frequency for a class label at a given shaft
    speed, or None for `Normal` (no defect, no characteristic frequency).
    """
    if label == "Normal":
        return None
    fault_type = label.rstrip("0123456789")
    fn = _FAULT_TYPE_TO_FREQUENCY_FN.get(fault_type)
    if fn is None:
        raise ValueError(f"Unrecognized fault label: {label!r}")
    return fn(rpm)
