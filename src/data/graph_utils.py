"""Turns paired DE/FE vibration windows into spatio-temporal graphs.

Each window becomes a small "ladder" graph: split each channel's window into
`n_segments` time segments, one node per (channel, segment). Edges connect
consecutive segments within a channel (temporal structure) and same-time
segments across channels (sensor coupling - DE and FE both ride the same
shaft assembly, so a real fault should show correlated disturbance in both).
Node features are the same per-segment time/frequency/envelope-spectrum
statistics used for the Phase 1 classical baseline (src/data/features.py),
not raw samples, so the graph structure - not per-node signal amplitude - is
what a GCN has to exploit; that's the actual thing Phase 3 is testing.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.cwru import SAMPLE_RATE_HZ
from src.data.features import extract_features

N_NODE_FEATURES = 21  # len(extract_features(...)): time + frequency + envelope stats


def build_edge_index(n_segments: int, n_channels: int = 2) -> torch.Tensor:
    """Ladder graph: n_channels rows x n_segments columns.

    Node id = channel_idx * n_segments + segment_idx.
    """
    edges = []
    for ch in range(n_channels):
        base = ch * n_segments
        for seg in range(n_segments - 1):
            edges.append((base + seg, base + seg + 1))
            edges.append((base + seg + 1, base + seg))  # temporal, both directions
    for seg in range(n_segments):
        for ch in range(n_channels - 1):
            a = ch * n_segments + seg
            b = (ch + 1) * n_segments + seg
            edges.append((a, b))
            edges.append((b, a))  # cross-channel coupling, both directions
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _segment_features(window: np.ndarray, n_segments: int, fs: float = SAMPLE_RATE_HZ) -> np.ndarray:
    segments = np.array_split(window, n_segments)
    return np.stack([list(extract_features(seg, fs).values()) for seg in segments])


def window_to_node_features(de_window: np.ndarray, fe_window: np.ndarray, n_segments: int) -> np.ndarray:
    """(2 * n_segments, N_NODE_FEATURES) - DE segments first, then FE segments."""
    return np.concatenate(
        [_segment_features(de_window, n_segments), _segment_features(fe_window, n_segments)], axis=0
    )


def build_graph_dataset(
    de_windows: np.ndarray,
    fe_windows: np.ndarray,
    targets: np.ndarray,
    n_segments: int = 8,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> list[Data]:
    """`feature_mean`/`feature_std` (per node-feature dim) standardize node
    features - node features span very different scales (e.g. td_rms ~0.1 vs
    td_kurtosis ~5), same issue that hurt un-scaled SVM in Phase 1. Compute
    them on the train set only (`compute_node_feature_stats`) and reuse for
    val/test, never fit on data a split shouldn't see.
    """
    edge_index = build_edge_index(n_segments)
    graphs = []
    for de_w, fe_w, y in zip(de_windows, fe_windows, targets):
        node_features = window_to_node_features(de_w, fe_w, n_segments)
        if feature_mean is not None:
            node_features = (node_features - feature_mean) / feature_std
        x = torch.tensor(node_features, dtype=torch.float32)
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long)))
    return graphs


def compute_node_feature_stats(
    de_windows: np.ndarray, fe_windows: np.ndarray, n_segments: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    all_features = np.concatenate(
        [window_to_node_features(de_w, fe_w, n_segments) for de_w, fe_w in zip(de_windows, fe_windows)],
        axis=0,
    )
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    std[std == 0] = 1.0
    return mean, std
