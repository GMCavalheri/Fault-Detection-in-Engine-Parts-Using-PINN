# Phase 2 — Deep Learning Results

Phase 1's classical baseline hit ~98% accuracy on the load-0-only 10-class problem — not a hard enough target for a meaningful deep-learning comparison. Phase 2 expanded CWRU to **full cross-condition coverage** (all 10 classes × all 4 loads, see [docs/datasets.md](datasets.md)) and used a genuine held-out-load split throughout: **train on loads 0–2, test entirely on the unseen load 3**.

Pipeline: `src/data/cwru.py` (unchanged) → `src/data/torch_utils.py` (global train-set standardization, not per-window — preserves cross-class amplitude differences) → `src/models/cnn.py` (1D CNN, raw signal) and `src/models/autoencoder.py` (1D conv autoencoder, unsupervised). Classical baseline from Phase 1 (`src/models/baseline.py`) retrained on the identical split for a fair comparison. Full run in [notebooks/02_deep_learning.ipynb](../notebooks/02_deep_learning.ipynb).

## Results summary

| Model | Task | Metric | Value |
|---|---|---|---|
| Random Forest (classical) | 10-class classification | accuracy | **0.978** |
| SVM, RBF (classical) | 10-class classification | accuracy | 0.831 |
| 1D CNN (deep learning) | 10-class classification | accuracy | 0.918 |
| Conv Autoencoder (deep learning) | binary anomaly detection | AUROC | 1.000 |

See [docs/phase2_confusion_matrices.png](phase2_confusion_matrices.png) and [docs/phase2_autoencoder_roc.png](phase2_autoencoder_roc.png).

## Finding 1: hand-engineered features + Random Forest generalize best across operating conditions

Random Forest barely drops from Phase 1's 0.977 (load-0-only) to 0.978 (cross-load) — the 21 hand-engineered features (time/frequency/envelope-spectrum statistics) turn out to already be close to load-invariant. This is the strongest, most robust result of the phase.

## Finding 2: the 1D CNN overfits loads 0–2 and collapses on one class at test time

The CNN hits 100% train **and** validation accuracy by epoch ~25 — a red flag for overfitting, confirmed by the held-out-load results: 0.918 accuracy overall, but **B021 (ball fault, 0.021" diameter) is never predicted at all** — every true B021 window at load 3 is misclassified as B014. With only 1,853 training windows and no data augmentation, the raw-signal CNN latched onto load-0–2-specific characteristics that happen to separate B014/B021 within those loads but don't transfer to load 3. Classical features avoid this because they're hand-designed to summarize signal *shape* (RMS, kurtosis, spectral centroid, etc.) rather than memorize raw waveform detail.

**Takeaway for the portfolio narrative:** more data and/or regularization (dropout, augmentation) would likely close this gap — worth revisiting if Phase 4/5 budget allows, but the more interesting story is that domain-informed features beat raw deep learning here specifically *because* the dataset is small (a genuinely common real-world finding, not a specific implementation bug).

## Finding 3: SVM (fixed hyperparameters) generalizes worst of the three

SVM drops sharply from 0.977 (load-0-only, Phase 1) to 0.831 (cross-load) — confusing B014/B021/OR014 heavily (e.g. IR014 recall drops to 0.15). Same features as Random Forest, same scaling fix from Phase 1, but a fixed `C=10, gamma="scale"` doesn't transfer across operating conditions as well as an ensemble of trees does. This is a natural, concrete motivation for **Phase 4** (Bayesian Optimization / PSO hyperparameter tuning) — SVM is the model most likely to benefit visibly from proper tuning rather than fixed defaults.

## Finding 4: the autoencoder ranks anomalies perfectly, but its threshold doesn't transfer across loads

AUROC = 1.000: `Normal`-load-3 reconstruction error (mean 0.041, max 0.051) and fault error (min 0.914) sit roughly 20x apart — by rank, separation is perfect. But fixed-threshold **detection accuracy is only 0.805**. Root cause (see the notebook's own load-3-normal vs. threshold callout): the threshold was calibrated as the 95th percentile of `Normal` windows from the *train* loads (0–2), and load-3's own `Normal` baseline reconstruction error runs slightly higher across its whole distribution (mean 0.041 > threshold 0.0395) — a small covariate shift across operating conditions pushes roughly half of the held-out-load `Normal` windows just over a threshold tuned on a different load.

This is a realistic problem for unsupervised deployment, not a bug: you can't peek at test-time fault examples to recalibrate, only at your own `Normal` validation data, and that data may not represent every future operating condition. Worth reopening with either a per-load-adaptive threshold or by feeding load/RPM as an auxiliary input (Phase 4/5).

## Takeaways for later phases

- Random Forest is the strongest baseline so far across every framing tried (load-0 classification, cross-load classification) — any future model needs to beat 0.978 accuracy to justify its complexity.
- SVM's cross-load degradation is a clean, concrete Phase 4 (hyperparameter optimization) target.
- The CNN's B021 collapse suggests small-dataset deep learning needs either more data (could down-sample windows more aggressively for more training examples, or gather more CWRU load/fault combinations) or regularization before it's a fair fight with classical features.
- The autoencoder's threshold-transfer problem is a good Phase 5 (physics-informed) hook: a threshold or reconstruction target that's explicitly conditioned on load/RPM (a known physical quantity, not something the model has to infer) could close this gap directly.
