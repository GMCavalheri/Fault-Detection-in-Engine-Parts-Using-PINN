# Phase 4 — Optimization Comparison Results

Tuned both models flagged as needing it: **SVM** (fixed hyperparameters degraded sharply under cross-load generalization, Phase 2's 0.831) and the **1D CNN** (the literal "best-performing model from Phases 2–3" per the project plan, 0.918). Optuna (Bayesian/TPE) and PSO (pyswarms) tuned the *same* continuous search space via the *same* objective function (`src/optimization/tuning.py`, `src/optimization/objectives.py`), scored on an inner validation split carved out of loads 0–2 — never the held-out load-3 test set. Full run in [notebooks/04_optimization.ipynb](../notebooks/04_optimization.ipynb).

## A methodology bug caught mid-phase, worth recording

The first run of the final "fixed vs. Optuna-tuned vs. PSO-tuned" CNN comparison reported the *fixed*-hyperparameter config (identical to Phase 2's) scoring a suspicious **1.0000** accuracy on held-out load 3 — nothing in three phases of work had ever gotten near that, and Phase 2's own number for the exact same hyperparameters was 0.918. Root cause: `torch.manual_seed(42)` was only set once at the top of the notebook, so the three final CNN retrains (fixed/Optuna/PSO) each inherited whatever random state was left over after the *preceding* tuning search had already consumed random numbers — meaning apparent differences between configs were partly just random-init/shuffling luck, not the hyperparameters themselves. Fix: reset the seed immediately before each of the three final retrains, isolating the variable actually being compared. Post-fix, the fixed config scored a believable 0.930 (see below) — this is the same kind of correctness check that caught the SVM scaling bug in Phase 1 and the histogram-binning bug in Phase 2.

## Results summary (held-out load 3)

| Model | Accuracy |
|---|---|
| **1D CNN (Optuna-tuned)** | **0.999** |
| **1D CNN (PSO-tuned)** | **0.999** |
| 1D CNN (fixed, Phase 2 hyperparameters) | 0.930 |
| Random Forest (classical, Phase 1/2) | 0.978 |
| SVM (PSO-tuned) | 0.836 |
| SVM (fixed, Phase 2 hyperparameters) | 0.831 |
| SVM (Optuna-tuned) | 0.826 |

**Both tuned CNN configs now beat Random Forest** — the first time any model has done so across four phases. See [docs/phase4_cnn_convergence.png](phase4_cnn_convergence.png) and [docs/phase4_svm_convergence.png](phase4_svm_convergence.png).

## Finding 1: tuning the CNN closed almost the entire gap to Random Forest — and then some

Optuna found `lr=0.0030, dropout=0.094, width_mult=0.73`; PSO independently found `lr=0.0012, dropout=0.073, width_mult=1.24` — different points in the search space, but both landed in the same neighborhood (moderate learning rate, small-but-nonzero dropout ~0.07–0.09) and both retrained to ~0.999 test accuracy. The fixed Phase 2 config used `dropout=0.0` — Phase 2's own write-up flagged the CNN as overfitting loads 0–2 (100% train/val accuracy, but a complete B021 collapse on held-out load 3). Adding even a small amount of dropout, found automatically by both optimizers independently, appears to have fixed exactly that overfitting problem. This is a clean, satisfying resolution to a problem identified two phases ago.

**Caveat:** this is a single seeded run per configuration, not averaged over multiple seeds — a genuinely rigorous claim would run each winning config 3–5 times and report a mean ± std. Treat 0.999 as "tuning clearly helps a lot," not as a guaranteed, exact number.

## Finding 2: SVM tuning didn't reliably help — validation accuracy plateaus faster than it should

Both optimizers converge to ~0.997 *validation* accuracy (loads 0–2) within 1–3 evaluations out of 40 — the 2D (C, gamma) space is easy enough that this comparison barely exercises either algorithm's actual search behavior. But that near-perfect validation score doesn't reliably transfer: PSO's winning config improved held-out-load accuracy only marginally (0.831 → 0.836), while Optuna's winning config was slightly *worse* (0.831 → 0.826) than the untuned default. The inner validation split (still loads 0–2, just held out in time) apparently isn't a good enough proxy for genuine cross-load generalization for this model — a different, harder kind of overfitting than the CNN's: not to the training *data*, but to the training *operating conditions*.

## Finding 3: for this problem size, Optuna vs. PSO barely matters

For both SVM and CNN, both optimizers saturate their validation objective within the first few evaluations (see the convergence plots — both curves are flat almost immediately). Wall-clock cost was comparable (SVM: Optuna 3.9s vs. PSO 3.0s for 40 evals each; CNN: Optuna 75s vs. PSO 86s for 15 evals each). Neither search space here is hard enough, or the compute budget generous enough, to show a meaningful convergence-speed gap between Bayesian optimization and PSO — that comparison would need either a harder/higher-dimensional space or tighter evaluation budgets to be discriminating. Worth remembering for Project 4 (the dedicated optimizer benchmark): this project's search spaces were too easy to be a good testbed for that specific question.

## Takeaways for later phases

- The CNN is now the strongest model (0.999, tuned), reversing three phases of "Random Forest wins." Phase 5's physics-informed extension should be benchmarked against the *tuned* CNN, not the Phase 2 fixed-hyperparameter one.
- SVM's cross-load weakness (flagged in Phase 2/3) is **not** simply a hyperparameter problem — tuning against an in-distribution validation set didn't close it. Whatever's wrong with SVM here is more fundamental (kernel choice, feature interaction) than `C`/`gamma`.
- If Project 4 (the dedicated GA vs. Bayesian vs. PSO vs. CMA-ES benchmark) reuses this project's models, it'll need harder search spaces than the ones used here to produce a meaningful convergence-speed comparison.
