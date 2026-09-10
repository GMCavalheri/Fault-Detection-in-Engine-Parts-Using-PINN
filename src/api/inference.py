"""Loads the trained artifacts (scripts/train_artifacts.py) once and exposes
plain-Python prediction functions - shared by the FastAPI app (src/api/main.py)
and reusable directly from a notebook or test.
"""

import json
from pathlib import Path

import numpy as np
import torch

from src.models.autoencoder import ConvAutoencoder
from src.models.cnn import Cnn1D
from src.physics.bearing import expected_fault_frequency_hz

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class FaultDetector:
    """Loads both artifacts on construction; call `.classify()` /
    `.anomaly_score()` / `.analyze()` per window after that.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        cnn_meta = json.loads((models_dir / "cnn_classifier_metadata.json").read_text())
        self.class_names: list[str] = cnn_meta["class_names"]
        self.window_size: int = cnn_meta["window_size"]
        self.sample_rate_hz: float = cnn_meta["sample_rate_hz"]
        self.cnn_train_mean: float = cnn_meta["train_mean"]
        self.cnn_train_std: float = cnn_meta["train_std"]

        self.cnn = Cnn1D(
            num_classes=len(self.class_names), width_mult=cnn_meta["width_mult"], dropout=cnn_meta["dropout"]
        )
        self.cnn.load_state_dict(torch.load(models_dir / "cnn_classifier.pt", map_location="cpu"))
        self.cnn.eval()

        ae_meta = json.loads((models_dir / "autoencoder_metadata.json").read_text())
        self.ae_train_mean: float = ae_meta["train_mean"]
        self.ae_train_std: float = ae_meta["train_std"]
        self.anomaly_threshold: float = ae_meta["anomaly_threshold_mse"]

        self.autoencoder = ConvAutoencoder()
        self.autoencoder.load_state_dict(torch.load(models_dir / "autoencoder.pt", map_location="cpu"))
        self.autoencoder.eval()

    @torch.no_grad()
    def classify(self, window: np.ndarray) -> dict:
        """Returns the predicted class, per-class confidence, and the raw
        softmax distribution for a single (window_size,) signal window.
        """
        x = (window - self.cnn_train_mean) / self.cnn_train_std
        x_tensor = torch.tensor(x, dtype=torch.float32).view(1, 1, -1)
        logits = self.cnn(x_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        pred_idx = int(np.argmax(probs))
        return {
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "class_probabilities": {name: float(p) for name, p in zip(self.class_names, probs)},
        }

    @torch.no_grad()
    def anomaly_score(self, window: np.ndarray) -> dict:
        """Reconstruction-error-based anomaly score for a single window."""
        x = (window - self.ae_train_mean) / self.ae_train_std
        x_tensor = torch.tensor(x, dtype=torch.float32).view(1, 1, -1)
        recon = self.autoencoder(x_tensor)
        mse = float(((recon - x_tensor) ** 2).mean())
        return {
            "reconstruction_error": mse,
            "anomaly_threshold": self.anomaly_threshold,
            "is_anomaly": mse > self.anomaly_threshold,
        }

    def physics_context(self, predicted_class: str, rpm: float | None) -> dict | None:
        """Optional Phase 5 context: the physically-expected defect frequency
        for the predicted fault class at a user-supplied RPM. None if no RPM
        was given or the predicted class has no characteristic frequency
        (Normal).
        """
        if rpm is None:
            return None
        freq_hz = expected_fault_frequency_hz(predicted_class, rpm)
        if freq_hz is None:
            return None
        return {"rpm": rpm, "expected_defect_frequency_hz": freq_hz}

    def analyze(self, window: np.ndarray, rpm: float | None = None) -> dict:
        """Full prediction: classification + confidence + anomaly score,
        plus optional physics context if RPM is supplied - the "prediction +
        confidence + anomaly score" the project plan's Phase 6 dashboard asks for.
        """
        if window.shape != (self.window_size,):
            raise ValueError(f"Expected a window of shape ({self.window_size},), got {window.shape}")

        result = self.classify(window)
        result.update(self.anomaly_score(window))
        result["physics_context"] = self.physics_context(result["predicted_class"], rpm)
        return result
