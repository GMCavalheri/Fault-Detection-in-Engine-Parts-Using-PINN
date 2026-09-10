"""1D CNN with an auxiliary physics-informed regression head.

Same conv trunk as `src/models/cnn.py::Cnn1D` (kept separate so Phases 1-4
stay exactly reproducible), plus a second head that predicts the window's
bearing defect frequency (src/physics/bearing.py) from the pooled features.
The auxiliary target comes from a physical equation (shaft speed + bearing
geometry), not a learned label, so supervising it is a genuine soft physics
constraint: the model is pushed toward a representation that tracks a known
physical relationship, not just whatever correlates with the class label.

`Normal` windows have no defect frequency - `physics_mask` (see
src/data/torch_utils.py::PhysicsWindowDataset) excludes them from the
auxiliary loss without needing a sentinel target value.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class PhysicsCnn1D(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 1.0, dropout: float = 0.0):
        super().__init__()
        c1, c2, c3 = (max(4, round(base * width_mult)) for base in (16, 32, 64))
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c3, num_classes)
        self.physics_head = nn.Linear(c3, 1)

    def forward(self, x):
        features = self.dropout(self.features(x).squeeze(-1))  # (batch, c3)
        return self.classifier(features), self.physics_head(features).squeeze(-1)


def train_physics_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    physics_weight: float = 0.0,
    device: str = "cpu",
) -> list[dict]:
    """`physics_weight=0.0` reduces to plain classification training - the
    fair baseline to compare against, using the exact same code path.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_criterion = nn.CrossEntropyLoss()
    physics_criterion = nn.MSELoss(reduction="none")

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y, physics_target, physics_mask in train_loader:
            x, y = x.to(device), y.to(device)
            physics_target, physics_mask = physics_target.to(device), physics_mask.to(device)

            optimizer.zero_grad()
            class_logits, physics_pred = model(x)
            class_loss = class_criterion(class_logits, y)

            if physics_mask.sum() > 0:
                per_sample_mse = physics_criterion(physics_pred, physics_target)
                physics_loss = (per_sample_mse * physics_mask).sum() / physics_mask.sum()
            else:
                physics_loss = torch.tensor(0.0, device=device)

            loss = class_loss + physics_weight * physics_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += (class_logits.argmax(1) == y).sum().item()
            train_total += len(y)

        val_loss, val_correct, val_total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for x, y, physics_target, physics_mask in val_loader:
                x, y = x.to(device), y.to(device)
                physics_target, physics_mask = physics_target.to(device), physics_mask.to(device)
                class_logits, physics_pred = model(x)
                class_loss = class_criterion(class_logits, y)
                if physics_mask.sum() > 0:
                    per_sample_mse = physics_criterion(physics_pred, physics_target)
                    physics_loss = (per_sample_mse * physics_mask).sum() / physics_mask.sum()
                else:
                    physics_loss = torch.tensor(0.0, device=device)
                loss = class_loss + physics_weight * physics_loss

                val_loss += loss.item() * len(y)
                val_correct += (class_logits.argmax(1) == y).sum().item()
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
def predict(model: nn.Module, loader: DataLoader, device: str = "cpu") -> dict:
    model.eval()
    model.to(device)
    class_preds, targets, physics_preds, physics_targets, physics_masks = [], [], [], [], []
    for x, y, physics_target, physics_mask in loader:
        class_logits, physics_pred = model(x.to(device))
        class_preds.append(class_logits.argmax(1).cpu().numpy())
        targets.append(y.numpy())
        physics_preds.append(physics_pred.cpu().numpy())
        physics_targets.append(physics_target.numpy())
        physics_masks.append(physics_mask.numpy())
    return {
        "class_preds": np.concatenate(class_preds),
        "targets": np.concatenate(targets),
        "physics_preds": np.concatenate(physics_preds),
        "physics_targets": np.concatenate(physics_targets),
        "physics_masks": np.concatenate(physics_masks),
    }
