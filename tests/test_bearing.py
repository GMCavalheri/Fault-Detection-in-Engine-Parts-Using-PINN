import pytest

from src.physics.bearing import bpfi_hz, bpfo_hz, bsf_hz, expected_fault_frequency_hz, ftf_hz

RPM = 1797  # CWRU load-0 shaft speed
F_R = RPM / 60  # 29.95 Hz


def test_defect_frequency_multipliers_match_published_cwru_values():
    # published CWRU/SKF-6205 multipliers (orders of shaft speed), see src/physics/bearing.py
    assert bpfo_hz(RPM) / F_R == pytest.approx(3.5848, rel=1e-3)
    assert bpfi_hz(RPM) / F_R == pytest.approx(5.4152, rel=1e-3)
    assert bsf_hz(RPM) / F_R == pytest.approx(2.357, rel=1e-3)
    assert ftf_hz(RPM) / F_R == pytest.approx(0.3983, rel=1e-3)


def test_frequencies_scale_linearly_with_rpm():
    # doubling shaft speed must exactly double every defect frequency
    assert bpfo_hz(2 * RPM) == pytest.approx(2 * bpfo_hz(RPM))
    assert bpfi_hz(2 * RPM) == pytest.approx(2 * bpfi_hz(RPM))
    assert bsf_hz(2 * RPM) == pytest.approx(2 * bsf_hz(RPM))


def test_inner_race_frequency_exceeds_outer_race_for_same_bearing():
    # true for any deep-groove ball bearing: BPFI > BPFO always (more balls
    # pass a fixed point on the inner race per revolution than the outer)
    assert bpfi_hz(RPM) > bpfo_hz(RPM)


def test_expected_fault_frequency_dispatches_by_label():
    assert expected_fault_frequency_hz("IR007", RPM) == pytest.approx(bpfi_hz(RPM))
    assert expected_fault_frequency_hz("IR021", RPM) == pytest.approx(bpfi_hz(RPM))
    assert expected_fault_frequency_hz("B014", RPM) == pytest.approx(bsf_hz(RPM))
    assert expected_fault_frequency_hz("OR007", RPM) == pytest.approx(bpfo_hz(RPM))
    assert expected_fault_frequency_hz("Normal", RPM) is None


def test_expected_fault_frequency_rejects_unknown_label():
    with pytest.raises(ValueError):
        expected_fault_frequency_hz("XYZ123", RPM)
