"""1D convolutional autoencoder for unsupervised anomaly detection.

Trained only on Normal-condition windows; at inference time, faulty windows
should reconstruct poorly (higher MSE) since the model never learned their
shape. Reconstruction error becomes the anomaly score.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=16, stride=4, padding=6),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=16, stride=4, padding=6),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    device: str = "cpu",
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
            total_n += len(x)
        history.append({"epoch": epoch, "train_loss": total_loss / total_n})
    return history


@torch.no_grad()
def reconstruction_errors(model: nn.Module, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    model.eval()
    model.to(device)
    errors = []
    for x, _ in loader:
        x = x.to(device)
        recon = model(x)
        per_window_mse = ((recon - x) ** 2).mean(dim=[1, 2])
        errors.append(per_window_mse.cpu().numpy())
    return np.concatenate(errors)
