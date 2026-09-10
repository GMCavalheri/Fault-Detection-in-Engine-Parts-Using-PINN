# Phase 3 — Graph Neural Network Results

Phases 1–2 only used the drive-end (DE) accelerometer channel. CWRU also records a fan-end (FE) channel on every file, including `Normal` (see [docs/datasets.md](datasets.md)), so Phase 3 builds a genuine multi-sensor graph per window instead of the simulated-topology fallback the project plan allowed for:

- **Nodes**: each 2048-sample window is split into 8 time segments per channel → 16 nodes (2 channels × 8 segments).
- **Edges**: temporal (segment *i* ↔ *i+1* within a channel) + cross-channel coupling (DE segment *i* ↔ FE segment *i*, same instant, since both accelerometers ride the same shaft assembly).
- **Model**: 3-layer GCN (`src/models/gnn.py`) + global mean pooling + linear classifier head.
- **Split**: identical held-out-load protocol as Phase 2 (train loads 0–2, test load 3), so results are directly comparable.

Pipeline: `src/data/cwru.py::build_multichannel_windowed_dataset` (DE+FE aligned windows) → `src/data/graph_utils.py` (window → per-segment-feature graph) → `src/models/gnn.py`. Full run in [notebooks/03_gnn.ipynb](../notebooks/03_gnn.ipynb).

## Results summary (held-out load 3)

| Model | Accuracy |
|---|---|
| Random Forest (classical, Phase 1/2) | **0.978** |
| 1D CNN (deep learning, Phase 2) | 0.918 |
| **GCN, DE+FE spatio-temporal graph (Phase 3)** | **0.871** |
| SVM, RBF (classical, Phase 2) | 0.831 |

See [docs/phase3_confusion_matrix.png](phase3_confusion_matrix.png).

## Finding 1: node feature richness mattered more than the graph structure

The first GCN run used only the 10 time-domain statistics (mean, RMS, kurtosis, etc.) per node — the same features that hurt un-scaled SVM in Phase 1 minus the frequency/envelope-spectrum half. That version scored **0.818** accuracy. Swapping in the full 21-feature set (adding frequency-domain and envelope-spectrum statistics per segment, `src/data/features.py::extract_features`) raised it to **0.871** — a +5.3 point jump from strictly more informative nodes, no architecture change. Lesson carried over from Phase 1: the envelope-spectrum features specifically are diagnostic for bearing faults, and any model — classical or graph-based — that's denied them pays for it.

## Finding 2: B021 (and OR014) are hard for every model except Random Forest

Across Phases 1–3, one class consistently causes trouble under cross-load generalization: **B021** (0.021" ball fault). Random Forest is the only model that classifies it reliably (Phase 2: recall 1.00). Every other model stumbles on it in a different way:
- CNN (Phase 2): B021 recall **0.00** — every held-out-load B021 window misclassified as B014.
- SVM (Phase 2): B021 recall 0.42, heavily confused with B014.
- GCN (Phase 3): B021 recall 0.34, confused across B007/B014/OR014.

**OR014** shows the same pattern (RF recall 0.88 vs. GCN 0.37, SVM 0.24 in earlier runs). The common thread: every model *except* Random Forest works from either raw signal (CNN) or short 256-sample segments (GCN nodes) or the same whole-window features with less robust hyperparameters (SVM). Random Forest is the only one that both (a) sees the full 2048-sample window's summary statistics at once and (b) uses an ensemble that's inherently more tolerant of a few noisy features. That combination seems to be what actually generalizes across operating conditions for this specific fault-severity level — not model sophistication.

## Finding 3: the GCN overfits similarly to the CNN, but recovers more via features than architecture

Train/val accuracy reaches ~99% by epoch 30 (up from ~92%/93% in the weaker time-domain-only run) while test accuracy sits at 0.871 — a real generalization gap, structurally similar to the CNN's 100%/100%/0.918 pattern in Phase 2. Unlike the CNN, richer input features closed part of the gap without touching model capacity or regularization, suggesting the GCN's bottleneck is currently more about *what it's given to work with* than the 3-layer/32-hidden-dim architecture itself.

## Takeaways for later phases

- Confirmed for the third time (RF in Phase 1, RF vs. SVM in Phase 2, GCN here): the frequency-domain and envelope-spectrum features are load-invariant discriminators, and their absence is the single most reliable predictor of a model doing badly. Any future model — Phase 5's physics-informed extension included — should keep or extend this feature family rather than dropping back to raw/time-domain-only signal.
- B021/OR014 cross-load generalization is now a recurring, well-documented weak spot for every non-Random-Forest model. Worth a dedicated look in Phase 4 (does hyperparameter tuning fix it for SVM/GCN?) rather than treating it as phase-specific noise.
- Random Forest remains undefeated across three phases and three very different modeling paradigms. Phase 4's optimization comparison (Optuna/PSO) should be judged partly by whether it can close this gap for SVM specifically, since that's the model most likely to benefit from proper tuning rather than fixed defaults.
