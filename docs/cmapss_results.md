# C-MAPSS RUL Regression Results

A follow-on beyond the original 6-phase plan: activates the second dataset acquired in Phase 0 but not modeled until now. Genuinely different task from the CWRU work — regression instead of classification, a multivariate low-frequency sensor time series instead of a raw high-frequency vibration signal, and a different evaluation protocol entirely (the field's own standard for this exact benchmark, not something invented for this project).

## Setup

FD001 subset (single operating condition, single fault mode, 100 train / 100 test engine trajectories) — the "starting subset" flagged as such back in [docs/datasets.md](datasets.md). Following the field's standard protocol for this benchmark (e.g. the PHM08 challenge, and the LSTM baseline from Zheng et al. 2017) so results are methodologically comparable to published work:

- **RUL capping at 125 cycles** for training labels (`src/data/cmapss.py::add_capped_rul`) — RUL is only learnable close to failure; a linear target from cycle 1 of a 300-cycle run is unlearnable noise.
- **30-cycle sliding windows** of sensor readings predict the RUL at the window's last cycle. Short trajectories are front-padded.
- **17 of 24 columns kept** (`select_informative_columns`, computed from training-set variance, not hardcoded) — FD001's single operating condition leaves 7 sensors and 1 operational setting exactly constant.
- **Unit-level train/val split** (85/15 units) — matches the leakage-aware split strategy in [docs/datasets.md](datasets.md); no cycle from a validation unit's trajectory appears in training.
- **Standard test protocol**: the dataset's own test trajectories are pre-truncated at an arbitrary point before failure; one prediction per unit from its last window, scored against `RUL_FD001.txt`'s true (uncapped) values.
- **Metrics**: RMSE (cycles) and the PHM08/C-MAPSS asymmetric scoring function (`src/models/rul_metrics.py::nasa_score`) — the field's standard metric, penalizing *late* predictions (overestimating remaining life, the operationally dangerous direction) far more heavily than early ones.

Two models, continuing the classical-vs-deep-learning comparison from Phases 1–2: Gradient Boosting on window-summary statistics (mean/std/last value per sensor), and an LSTM directly over the raw 30-cycle sequences. Full run in [notebooks/07_cmapss_rul_regression.ipynb](../notebooks/07_cmapss_rul_regression.ipynb).

## Results

| Model | Test RMSE (cycles) | NASA score |
|---|---|---|
| **Gradient Boosting (classical)** | **17.04** | **582.8** |
| LSTM (deep learning) | 18.89 | 991.4 |

The classical model wins on both metrics — decisively on the NASA score, which is the more operationally meaningful one. See [docs/cmapss_predicted_vs_true_rul.png](cmapss_predicted_vs_true_rul.png).

## Finding: the LSTM overfits, and the training curve shows exactly why

[docs/cmapss_lstm_training_curve.png](cmapss_lstm_training_curve.png) is a textbook overfitting curve: train RMSE keeps falling smoothly to ~5 cycles over 30 epochs, while validation RMSE plateaus around 14–18 by epoch 5 and drifts slightly *upward* after epoch ~20. The predicted-vs-true scatter plot shows the consequence directly: the LSTM has several points where a true RUL of 40–90 cycles gets predicted at 100–125 — a large, late-direction miss, which is exactly what the NASA score punishes hardest, explaining why its score (991) looks so much worse than its RMSE gap over Gradient Boosting alone would suggest (18.9 vs 17.0 is not dramatic; 991 vs 583 is).

**Why this happens**: FD001 has only 85 training-split trajectories. Windowing produces ~15,000 individual sequences, but they're heavily autocorrelated — consecutive windows from the same engine share nearly all their cycles. The LSTM's larger parameter count needs more *independent* degradation trajectories than that to generalize well; Gradient Boosting's built-in regularization (shallow trees, shrinkage, an ensemble of weak learners) handles this data-scarcity regime more gracefully. This mirrors a genuine, recognized practical concern in the RUL literature — deep sequence models on C-MAPSS-scale data often need explicit help (dropout, data augmentation, transfer learning across FD subsets) to match well-tuned classical baselines, and this project's result reproduces that pattern rather than being an implementation bug.

This is the same story as Phases 1–3 of the CWRU work, playing out again in a completely different task and dataset: a properly-built classical model is a genuinely hard baseline to beat, and "used a neural network" is not automatically better. Phase 4 showed hyperparameter tuning *can* close that gap for CWRU's CNN — the natural next experiment here would be the same treatment for this LSTM (dropout tuning in particular, given the overfitting signature) rather than accepting this result as the ceiling.

## Scope notes

- FD001 only. FD002 (six operating conditions) is now modeled too — see [docs/cmapss_fd002_results.md](cmapss_fd002_results.md), which needed its own regime-based normalization step that FD001 didn't. FD004 (six conditions *and* two fault modes) is downloaded but still unexplored.
- No physics-informed component here, unlike Phase 5's CWRU work. There is a real physics angle available (RUL should be monotonically non-increasing as cycles progress within a trajectory — a natural soft constraint) but it wasn't part of this task's scope; noting it rather than building it unasked.
- Not integrated into the Phase 6 API/dashboard. Would need its own endpoint (`/predict_rul`, taking a sequence of cycles rather than a single vibration window) — a reasonable follow-on if this becomes a second served model.
