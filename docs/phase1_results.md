# Phase 1 — Classical Baseline Results

Pipeline: `src/data/cwru.py` (load + non-overlapping 2048-sample windows, 12 kHz → ~170 ms/window) → `src/data/features.py` (21 hand-engineered features: 10 time-domain, 5 frequency-domain, 6 envelope-spectrum) → `src/models/baseline.py` (Random Forest, SVM with RBF kernel). Full run in [notebooks/01_classical_baseline.ipynb](../notebooks/01_classical_baseline.ipynb).

## Primary evaluation: 10-class fault classification (load 0)

1,157 windows across 10 classes (Normal, IR/B/OR × 007/014/021 fault diameters), all at load 0 (1797 rpm). Split: last 20% of each file's windows (in time order) held out for test — see `time_contiguous_split` in [src/models/baseline.py](../src/models/baseline.py) and the leakage rationale in [docs/datasets.md](datasets.md).

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest (200 trees) | 0.977 | 0.98 |
| SVM (RBF, scaled features) | 0.977 | 0.98 |

Both models confuse `OR014` and `B021` with `B014`/`B021` occasionally (visually adjacent fault-diameter classes) — see [docs/phase1_confusion_matrices.png](phase1_confusion_matrices.png). No systematic weakness beyond that; `Normal` and `IR*` classes are separated perfectly by both models.

**Important fix during this phase:** the SVM initially scored only 0.73 accuracy (vs. RF's 0.977) on the exact same features — not a real model gap, but unscaled features (e.g. `fd_energy` ~1e5 vs. `td_mean` ~1e-2) starving the RBF kernel's distance metric. Wrapping SVM in a `StandardScaler` pipeline brought it to parity with Random Forest. Worth remembering for any future distance-based model (k-NN, plain neural nets without batch norm) on these features.

## Leakage check: naive random split vs. time-contiguous split

Random Forest accuracy: **0.985** (naive random window split, ignoring source file) vs. **0.977** (time-contiguous split). The gap is small here — with 2048-sample non-overlapping windows the within-file correlation isn't huge — but it's in the expected direction (naive split reports higher, over-optimistic accuracy), which is exactly the leakage the split strategy in `docs/datasets.md` is designed to catch. Worth re-checking this gap once windows get smaller/overlapping or models get more expressive (deep learning phases) — leakage effects tend to grow with model capacity.

## Secondary evaluation: cross-load generalization (IR007 vs. Normal, held-out load)

Train on loads 0–2, test entirely on the held-out load-3 file (`IR007` and `Normal` were downloaded across all 4 loads specifically for this check — see [docs/datasets.md](datasets.md)). Random Forest: **1.00 accuracy** on both classes. For this binary case at least, the hand-engineered features are load-invariant — motor load changes shaft speed slightly (1797→1730 rpm) but doesn't shift the fault signature enough to fool the classifier. Worth re-testing this once more classes and load conditions are in the mix (Phase 2+).

## Takeaways for later phases

- 21 features and ~1,150 windows are enough for near-perfect classical classification on the load-0 10-class problem — deep learning (Phase 2) will need a harder framing to be a meaningful comparison: either the full 4-load cross-condition problem, unsupervised anomaly detection (autoencoder) rather than supervised classification, or C-MAPSS RUL regression instead of CWRU classification.
- Feature scaling matters a lot for non-tree models — apply the same lesson to any future baseline (k-NN, plain feedforward nets).
- The time-contiguous vs. random split gap is currently small; re-check it as model capacity grows.
