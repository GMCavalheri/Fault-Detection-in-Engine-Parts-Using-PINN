# C-MAPSS FD002 Results: Regime-Based Normalization Ablation

Extends the [FD001 RUL regression work](cmapss_results.md) to FD002 — 260 train / 259 test engine trajectories across **six operating conditions** instead of FD001's one. That difference changes what "correct preprocessing" means enough to warrant its own investigation rather than just re-running the FD001 pipeline unchanged. Full run in [notebooks/08_cmapss_fd002_regime_normalization.ipynb](../notebooks/08_cmapss_fd002_regime_normalization.ipynb).

## The problem: condition-conflation

Clustering `op_setting_1-3` confirms exactly 6 discrete operating regimes (matching the literature). Within FD002's training data, `sensor_2`'s mean varies from **536.9 to 642.7** depending on which regime an engine happens to be in — a ~106-unit swing — while its within-regime standard deviation is only ~0.4–0.5. A sensor reading's value is therefore driven far more by *which operating condition the engine is currently in* than by how degraded it is. FD001's pipeline (`select_informative_columns` + a single global mean/std, see [docs/cmapss_results.md](cmapss_results.md)) has no way to distinguish those two sources of variation — it was never a problem for FD001 because that dataset only has one condition.

## The fix

`src/data/cmapss.py::RegimeNormalizer` (new): k-means clusters `op_setting_1-3` into regimes (fit on train only, `k=6`), then z-scores each sensor **within its own regime** using train-fit per-regime statistics — regime assignment for held-out data uses nearest-centroid prediction from the training-fit clusters, so nothing about the test set influences the normalization. What's left after this step is much closer to pure degradation signal.

## Results: naive vs. regime-normalized, both models

| Pipeline | Model | Test RMSE (cycles) | NASA score |
|---|---|---|---|
| naive (FD001-style) | Gradient Boosting | 31.86 | 15,301.5 |
| naive (FD001-style) | LSTM | 31.65 | 36,962.6 |
| **regime-normalized** | **Gradient Boosting** | **29.64** | **14,419.7** |
| **regime-normalized** | **LSTM** | **29.63** | **17,195.9** |

Regime normalization helps **both** models on RMSE by a modest, consistent ~7% (31.9→29.6, 31.7→29.6) — a real but unremarkable improvement. The NASA score tells a much more dramatic story: it drops 5.8% for Gradient Boosting but **53.5%** for the LSTM (36,963 → 17,196, more than halving it). The asymmetric score is dominated by a handful of severe late-direction misses (a single prediction of RUL≈125 against a true RUL near 0 contributes over 260,000 to the sum on its own — `exp(12.5)-1`), so this result says the naive LSTM was occasionally catastrophically overconfident about remaining life for condition-conflated inputs, and regime normalization mostly (not entirely) fixes that specific failure mode. Gradient Boosting's tree-based splits were apparently more robust to the same conflation to begin with, so it had less room to improve — consistent with the same "classical model is a hard, robust baseline" pattern from every other phase of this project.

## FD002 vs. FD001: a much harder problem regardless of preprocessing

Even the best FD002 result (RMSE 29.6) is nearly **double** FD001's best (RMSE 17.0, [docs/cmapss_results.md](cmapss_results.md)), and FD002's NASA scores are two to three orders of magnitude larger (14,000+ vs. 583–991). This isn't a preprocessing artifact — it reflects FD002 genuinely being a harder prediction problem: 2.6x more engine trajectories, six operating regimes instead of one, and (per the dataset's own documentation) more heterogeneous degradation dynamics. Matches the universal pattern in published C-MAPSS work, where FD002/FD004 are consistently reported as substantially harder than FD001/FD003.

## Takeaways

- Regime-based normalization is a real, evidence-backed preprocessing requirement for FD002/FD004 specifically, not a hypothetical concern — confirmed with FD002's own data (the sensor_2 offset) before building it, and confirmed to help via a controlled naive-vs-regime ablation rather than assumed.
- Its biggest win is fixing the LSTM's worst-case failures (NASA score), not its average-case error (RMSE) — a good reminder that RMSE alone would have understated how much this preprocessing step matters here.
- FD002's absolute performance ceiling is much lower than FD001's under this project's current approach (fixed 30-cycle window, fixed architecture/hyperparameters reused from FD001 unchanged). The natural next step, if this gets picked up again, is hyperparameter tuning specifically for FD002 (mirroring Phase 4's Optuna/PSO treatment of the CWRU CNN) rather than assuming FD001's chosen settings transfer.
- FD004 (six conditions *and* two fault modes — the hardest C-MAPSS subset) is downloaded but still unexplored.
