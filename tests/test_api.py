import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, detector

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert len(body["classes"]) == 10
    assert body["window_size"] == detector.window_size


def _csv_bytes(n_samples: int) -> bytes:
    values = np.random.default_rng(0).normal(size=n_samples)
    buf = io.StringIO()
    np.savetxt(buf, values)
    return buf.getvalue().encode("utf-8")


def test_predict_accepts_csv_upload_and_returns_valid_response():
    files = {"file": ("signal.csv", _csv_bytes(detector.window_size), "text/csv")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 200
    body = resp.json()
    assert body["n_windows"] == 1
    assert body["predicted_class"] in detector.class_names
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["anomaly_rate"] <= 1.0
    assert len(body["windows"]) == 1


def test_predict_windows_a_longer_signal_into_multiple_predictions():
    files = {"file": ("signal.csv", _csv_bytes(detector.window_size * 3), "text/csv")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 200
    assert resp.json()["n_windows"] == 3


def test_predict_with_rpm_includes_physics_context_for_fault_predictions():
    files = {"file": ("signal.csv", _csv_bytes(detector.window_size), "text/csv")}
    resp = client.post("/predict", files=files, data={"rpm": 1797})

    assert resp.status_code == 200
    body = resp.json()
    window = body["windows"][0]
    if window["predicted_class"] != "Normal":
        assert window["physics_context"] is not None
        assert window["physics_context"]["rpm"] == 1797


def test_predict_rejects_signal_shorter_than_window_size():
    files = {"file": ("signal.csv", _csv_bytes(detector.window_size - 100), "text/csv")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 400


def test_predict_rejects_unsupported_file_type():
    files = {"file": ("signal.xyz", b"not a real signal", "application/octet-stream")}
    resp = client.post("/predict", files=files)

    assert resp.status_code == 400
