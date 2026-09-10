"""LSTM regressor for RUL prediction from sliding-window C-MAPSS sequences."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class LstmRul(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        # x: (batch, window_size, n_features)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # (batch, hidden_size) - the window's final timestep
        return self.head(last_hidden).squeeze(-1)


def train_lstm(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 30, lr: float = 1e-3,
    device: str = "cpu",
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        model.train()
        train_sse, train_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_sse += loss.item() * len(y)
            train_n += len(y)

        model.eval()
        val_sse, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_sse += criterion(pred, y).item() * len(y)
                val_n += len(y)

        history.append(
            {"epoch": epoch, "train_rmse": (train_sse / train_n) ** 0.5, "val_rmse": (val_sse / val_n) ** 0.5}
        )
    return history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    preds, targets = [], []
    for x, y in loader:
        pred = model(x.to(device))
        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)
