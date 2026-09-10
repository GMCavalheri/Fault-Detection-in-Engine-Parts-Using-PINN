"""GCN classifier over per-window DE/FE spatio-temporal graphs (src/data/graph_utils.py)."""

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class GcnClassifier(nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)  # graph-level embedding
        return self.classifier(x)


def train_gnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
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
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            train_correct += (logits.argmax(1) == batch.y).sum().item()
            train_total += batch.num_graphs

        val_loss, val_correct, val_total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                val_loss += loss.item() * batch.num_graphs
                val_correct += (logits.argmax(1) == batch.y).sum().item()
                val_total += batch.num_graphs

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
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)
