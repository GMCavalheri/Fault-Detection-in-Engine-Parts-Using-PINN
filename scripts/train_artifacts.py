"""Trains and saves the models/ artifacts the API and dashboard serve.

Standard practice: architecture and hyperparameters were already validated on
held-out splits in Phases 2-4 (the CNN hit 0.999 accuracy on a held-out-load
test using these exact hyperparameters). This script retrains on ALL
available labeled data for the actual deployed artifact, since there's no
more validation to do - just maximizing what the shipped model has seen.

Produces:
  models/cnn_classifier.pt + models/cnn_classifier_metadata.json
  models/autoencoder.pt + models/autoencoder_metadata.json

Run: uv run python scripts/train_artifacts.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.data.cwru import SAMPLE_RATE_HZ, build_windowed_dataset
from src.data.torch_utils import WindowDataset, compute_train_stats
from src.models.autoencoder import ConvAutoencoder, reconstruction_errors, train_autoencoder
from src.models.baseline import time_contiguous_split
from src.models.cnn import Cnn1D, predict, train_cnn

REPO_ROOT = Path(__file__).resolve().parent.parent
CWRU_DIR = REPO_ROOT / "data" / "raw" / "cwru"
MODELS_DIR = REPO_ROOT / "models"
WINDOW_SIZE = 2048

# Phase 4's Optuna-winning CNN hyperparameters (docs/phase4_results.md)
CNN_LR = 0.0030405325392865654
CNN_DROPOUT = 0.0936111842654619
CNN_WIDTH_MULT = 0.7339917805043033
CNN_EPOCHS = 30

AE_EPOCHS = 30
AE_THRESHOLD_PERCENTILE = 95


def train_classifier(windows: np.ndarray, labels: np.ndarray, file_ids: np.ndarray) -> None:
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_.tolist()

    # sanity-check split: not used to pick anything, just a printed reality check
    # that the same architecture/hyperparameters still fit this (larger, all-load) dataset well
    sanity_train_mask, sanity_val_mask = time_contiguous_split(file_ids, test_fraction=0.1)
    mean, std = compute_train_stats(windows[sanity_train_mask])
    sanity_train_loader = DataLoader(
        WindowDataset(windows[sanity_train_mask], y[sanity_train_mask], mean, std), batch_size=64, shuffle=True
    )
    sanity_val_loader = DataLoader(
        WindowDataset(windows[sanity_val_mask], y[sanity_val_mask], mean, std), batch_size=64
    )
    torch.manual_seed(42)
    sanity_model = Cnn1D(num_classes=len(class_names), width_mult=CNN_WIDTH_MULT, dropout=CNN_DROPOUT)
    train_cnn(sanity_model, sanity_train_loader, sanity_val_loader, epochs=CNN_EPOCHS, lr=CNN_LR)
    preds, targets = predict(sanity_model, sanity_val_loader)
    sanity_accuracy = float((preds == targets).mean())
    print(f"[cnn] sanity-check accuracy on a held-out 10% slice: {sanity_accuracy:.4f}")

    # final artifact: retrain on ALL data
    train_mean, train_std = compute_train_stats(windows)
    full_loader = DataLoader(WindowDataset(windows, y, train_mean, train_std), batch_size=64, shuffle=True)
    torch.manual_seed(42)
    final_model = Cnn1D(num_classes=len(class_names), width_mult=CNN_WIDTH_MULT, dropout=CNN_DROPOUT)
    train_cnn(final_model, full_loader, full_loader, epochs=CNN_EPOCHS, lr=CNN_LR)

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(final_model.state_dict(), MODELS_DIR / "cnn_classifier.pt")
    metadata = {
        "class_names": class_names,
        "window_size": WINDOW_SIZE,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "width_mult": CNN_WIDTH_MULT,
        "dropout": CNN_DROPOUT,
        "train_mean": train_mean,
        "train_std": train_std,
        "sanity_check_accuracy": sanity_accuracy,
        "trained_on": "all 40 CWRU files (all loads, all classes)",
    }
    (MODELS_DIR / "cnn_classifier_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[cnn] saved models/cnn_classifier.pt + metadata ({len(class_names)} classes)")


def train_anomaly_detector(windows: np.ndarray, labels: np.ndarray, file_ids: np.ndarray) -> None:
    normal_mask = labels == "Normal"
    normal_windows = windows[normal_mask]
    normal_file_ids = file_ids[normal_mask]

    # calibration split: train on 90%, compute the anomaly threshold from the held-out 10%
    # (same normal-only-data-shift caveat as Phase 2 - see docs/phase2_results.md finding 4)
    cal_train_mask, cal_val_mask = time_contiguous_split(normal_file_ids, test_fraction=0.1)
    ae_mean, ae_std = compute_train_stats(normal_windows[cal_train_mask])

    cal_train_loader = DataLoader(
        WindowDataset(normal_windows[cal_train_mask], np.zeros(cal_train_mask.sum()), ae_mean, ae_std),
        batch_size=64,
        shuffle=True,
    )
    cal_val_loader = DataLoader(
        WindowDataset(normal_windows[cal_val_mask], np.zeros(cal_val_mask.sum()), ae_mean, ae_std), batch_size=64
    )
    torch.manual_seed(42)
    cal_model = ConvAutoencoder()
    train_autoencoder(cal_model, cal_train_loader, epochs=AE_EPOCHS)
    val_errors = reconstruction_errors(cal_model, cal_val_loader)
    threshold = float(np.percentile(val_errors, AE_THRESHOLD_PERCENTILE))
    print(f"[autoencoder] anomaly threshold ({AE_THRESHOLD_PERCENTILE}th pct of held-out normal error): {threshold:.5f}")

    # final artifact: retrain on ALL normal windows
    full_mean, full_std = compute_train_stats(normal_windows)
    full_loader = DataLoader(
        WindowDataset(normal_windows, np.zeros(len(normal_windows)), full_mean, full_std), batch_size=64, shuffle=True
    )
    torch.manual_seed(42)
    final_model = ConvAutoencoder()
    train_autoencoder(final_model, full_loader, epochs=AE_EPOCHS)

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(final_model.state_dict(), MODELS_DIR / "autoencoder.pt")
    metadata = {
        "window_size": WINDOW_SIZE,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "train_mean": full_mean,
        "train_std": full_std,
        "anomaly_threshold_mse": threshold,
        "threshold_percentile": AE_THRESHOLD_PERCENTILE,
        "trained_on": "all Normal-class windows across all 4 loads",
    }
    (MODELS_DIR / "autoencoder_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[autoencoder] saved models/autoencoder.pt + metadata ({len(normal_windows)} normal windows)")


def main() -> None:
    print(f"Loading CWRU windows from {CWRU_DIR} ...")
    windows, labels, loads, file_ids = build_windowed_dataset(CWRU_DIR, window_size=WINDOW_SIZE)
    print(f"Loaded {len(windows)} windows across {len(set(labels))} classes, {len(set(loads))} loads.")

    train_classifier(windows, labels, file_ids)
    train_anomaly_detector(windows, labels, file_ids)


if __name__ == "__main__":
    main()
