"""1D CNN classifier for raw CWRU vibration windows.

Wide first-layer kernel (captures low-frequency structure directly, a la
WDCNN) followed by two narrow conv blocks and global average pooling, so the
classifier head doesn't depend on window length.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class Cnn1D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x).squeeze(-1)  # (batch, 64)
        return self.classifier(x)


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += len(y)

        val_loss, val_correct, val_total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += len(y)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / train_total,
                "train_acc": train_correct / train_total,
                "val_loss": val_loss / val_total,
                "val_acc": val_correct / val_total,
            }
        )
    return history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    all_preds, all_targets = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)
