import numpy as np

from src.data.graph_utils import (
    N_NODE_FEATURES,
    build_edge_index,
    build_graph_dataset,
    compute_node_feature_stats,
    window_to_node_features,
)

WINDOW_SIZE = 2048
N_SEGMENTS = 8


def test_build_edge_index_shape_and_bounds():
    edge_index = build_edge_index(n_segments=N_SEGMENTS, n_channels=2)

    # temporal: 2 channels * 7 adjacent pairs * 2 directions = 28
    # cross-channel: 8 segments * 1 channel pair * 2 directions = 16
    assert edge_index.shape == (2, 44)
    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < 2 * N_SEGMENTS


def test_window_to_node_features_shape():
    rng = np.random.default_rng(0)
    de_window = rng.normal(size=WINDOW_SIZE)
    fe_window = rng.normal(size=WINDOW_SIZE)

    features = window_to_node_features(de_window, fe_window, N_SEGMENTS)
    assert features.shape == (2 * N_SEGMENTS, N_NODE_FEATURES)
    assert np.all(np.isfinite(features))


def test_build_graph_dataset_and_standardization():
    rng = np.random.default_rng(0)
    de_windows = rng.normal(size=(5, WINDOW_SIZE))
    fe_windows = rng.normal(size=(5, WINDOW_SIZE))
    targets = np.array([0, 1, 0, 1, 0])

    mean, std = compute_node_feature_stats(de_windows, fe_windows, N_SEGMENTS)
    assert mean.shape == (N_NODE_FEATURES,)

    graphs = build_graph_dataset(de_windows, fe_windows, targets, N_SEGMENTS, mean, std)
    assert len(graphs) == 5
    g = graphs[0]
    assert g.x.shape == (2 * N_SEGMENTS, N_NODE_FEATURES)
    assert g.edge_index.shape[0] == 2
    assert g.y.item() in (0, 1)

    # standardized features across the (small) train set should be roughly centered
    all_x = np.concatenate([g.x.numpy() for g in graphs], axis=0)
    assert np.abs(all_x.mean()) < 1.0
