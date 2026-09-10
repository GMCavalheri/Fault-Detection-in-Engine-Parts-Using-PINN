import numpy as np
from torch_geometric.loader import DataLoader

from src.data.graph_utils import N_NODE_FEATURES, build_graph_dataset, compute_node_feature_stats
from src.models.gnn import GcnClassifier, predict, train_gnn

WINDOW_SIZE = 2048
N_SEGMENTS = 8


def _synthetic_loader(n_samples: int, n_classes: int) -> DataLoader:
    rng = np.random.default_rng(0)
    de_windows = rng.normal(size=(n_samples, WINDOW_SIZE))
    fe_windows = rng.normal(size=(n_samples, WINDOW_SIZE))
    targets = rng.integers(0, n_classes, size=n_samples)

    mean, std = compute_node_feature_stats(de_windows, fe_windows, N_SEGMENTS)
    graphs = build_graph_dataset(de_windows, fe_windows, targets, N_SEGMENTS, mean, std)
    return DataLoader(graphs, batch_size=4, shuffle=True)


def test_gcn_forward_pass_shape():
    loader = _synthetic_loader(n_samples=8, n_classes=3)
    model = GcnClassifier(num_node_features=N_NODE_FEATURES, num_classes=3)
    batch = next(iter(loader))
    logits = model(batch.x, batch.edge_index, batch.batch)
    assert logits.shape == (batch.num_graphs, 3)


def test_train_gnn_one_epoch_smoke_test():
    loader = _synthetic_loader(n_samples=20, n_classes=3)
    model = GcnClassifier(num_node_features=N_NODE_FEATURES, num_classes=3)

    history = train_gnn(model, loader, loader, epochs=1)
    assert len(history) == 1
    assert set(history[0]) == {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}

    preds, targets = predict(model, loader)
    assert preds.shape == targets.shape == (20,)
