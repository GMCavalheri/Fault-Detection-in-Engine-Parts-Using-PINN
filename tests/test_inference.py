import numpy as np
import pytest

from src.api.inference import FaultDetector


@pytest.fixture(scope="module")
def detector() -> FaultDetector:
    return FaultDetector()


def test_detector_loads_with_expected_classes(detector):
    assert len(detector.class_names) == 10
    assert "Normal" in detector.class_names
    assert detector.window_size == 2048


def test_classify_returns_valid_probability_distribution(detector):
    window = np.random.default_rng(0).normal(size=detector.window_size)
    result = detector.classify(window)

    assert result["predicted_class"] in detector.class_names
    probs = result["class_probabilities"]
    assert set(probs) == set(detector.class_names)
    assert pytest.approx(sum(probs.values()), abs=1e-4) == 1.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_anomaly_score_returns_nonnegative_error(detector):
    window = np.random.default_rng(0).normal(size=detector.window_size)
    result = detector.anomaly_score(window)

    assert result["reconstruction_error"] >= 0
    assert isinstance(result["is_anomaly"], bool)
    assert result["anomaly_threshold"] == detector.anomaly_threshold


def test_analyze_rejects_wrong_shape(detector):
    with pytest.raises(ValueError):
        detector.analyze(np.zeros(100))


def test_analyze_combines_classification_and_anomaly_score(detector):
    window = np.random.default_rng(1).normal(size=detector.window_size)
    result = detector.analyze(window)

    assert "predicted_class" in result
    assert "confidence" in result
    assert "reconstruction_error" in result
    assert "is_anomaly" in result


def test_physics_context_none_without_rpm(detector):
    assert detector.physics_context("IR007", rpm=None) is None


def test_physics_context_none_for_normal_even_with_rpm(detector):
    # Normal has no characteristic defect frequency - see src/physics/bearing.py
    assert detector.physics_context("Normal", rpm=1797) is None


def test_physics_context_populated_for_fault_class_with_rpm(detector):
    context = detector.physics_context("IR007", rpm=1797)
    assert context is not None
    assert context["rpm"] == 1797
    assert context["expected_defect_frequency_hz"] > 0
