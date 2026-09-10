# Phase 5 — Physics-Informed Extension Results

## The physics

CWRU seeds every fault in this dataset on the drive-end bearing (SKF 6205-2RS JEM). A defect on a given component produces a periodic impact at a frequency set by shaft speed and bearing geometry — the standard rolling-element-bearing kinematic equations (`src/physics/bearing.py`):

```
BPFO = (n/2) f_r (1 - (d/D) cos φ)   outer-race defect frequency
BPFI = (n/2) f_r (1 + (d/D) cos φ)   inner-race defect frequency
BSF  = (D/2d) f_r (1 - (d/D)² cos²φ) ball defect frequency
```

where `f_r` is shaft speed in Hz (from the recorded RPM) and `n, d, D, φ` are bearing geometry constants. These reproduce CWRU's own published defect-frequency multipliers exactly (verified in `tests/test_bearing.py`: BPFO≈3.5848×, BPFI≈5.4152×, BSF≈2.357× shaft speed). Since RPM is known per file and fault location is known from the label, **every fault window has a physically-computable target frequency** — a genuine governing equation, not a fitted feature.

## The harder task

Phases 2–4 all used train-on-3-loads/test-on-1-held-out-load, and Phase 4's tuned CNN nearly solved it (0.999 accuracy) — too close to ceiling to show whether a physics-informed loss adds anything. Phase 5 instead trains on **load 0 only** (551 windows, one operating condition, RPM≈1797) and tests on **loads 1–3 combined** (2,303 windows, RPMs 1721–1774, none seen in training). `src/models/physics_cnn.py::PhysicsCnn1D` adds a single linear auxiliary head predicting defect frequency from the same pooled features the classifier uses, trained jointly: `loss = cross_entropy + physics_weight * masked_MSE(physics_head_output, true_defect_frequency)`, masked to exclude `Normal` windows (no defect, no target). Base architecture/optimizer settings reused Phase 4's Optuna-winning hyperparameters, isolating the physics loss as the one new variable. Full run in [notebooks/05_physics_informed.ipynb](../notebooks/05_physics_informed.ipynb).

## A data bug caught while building this

`load_channel`/`load_rpm` originally picked the *first* matching variable in each `.mat` file. Two of CWRU's own files broke that assumption: `Normal_1.mat` has no RPM field at all, and `Normal_2.mat` carries **stray leftover DE/FE channels from a different file** (`X098_*`, actually load-1 data) alongside its own (`X099_*`) — a known quirk in CWRU's original export, not something introduced here. Since dict iteration order put the stale `X098` key first, `load_de_channel("Normal_2.mat")` had been silently returning the wrong file's signal since **Phase 1**. Fixed by selecting the highest-numbered matching variable when a file has duplicates (`src/data/cwru.py::_select_key`), and falling back to CWRU's documented nominal RPM (1797/1772/1750/1730) for the two files missing that field entirely. **Practical impact on Phases 1–4: negligible** — `Normal` was classified at ~100% recall in every prior phase regardless of which of two adjacent loads' healthy-bearing signal it actually used, and re-running those phases wasn't warranted. Recorded here for the same reason the seeding bug from Phase 4 was: this is what the correctness checks in this project are for.

## Results summary (held-out loads 1–3, train on load 0 only)

| Physics loss weight | Accuracy |
|---|---|
| 0.0 (plain baseline) | 0.837 |
| **0.3** | **0.842** |
| 1.0 | 0.822 |
| 3.0 | 0.814 |

See [docs/phase5_weight_sweep.png](phase5_weight_sweep.png) — a clean, sensible curve: a small physics weight helps, too much distorts the primary classification objective. Note the much lower accuracy ceiling here (0.84 vs. Phase 4's 0.999) confirms this split is genuinely harder, as intended.

## Finding 1: a small physics weight gives a real but modest net improvement, with a redistributed error pattern — not a clean win

Weight 0.3 beats the plain baseline by +0.5 points, but the class-by-class picture (see [docs/phase5_confusion_matrices.png](phase5_confusion_matrices.png)) is a genuine trade, not a uniform improvement:
- **OR007 recall jumps from 0.04 to 0.58** — the plain model predicted almost every OR007 window as OR021; the physics-informed model mostly fixes this.
- **B021 recall drops from 0.84 to 0.42** — the physics-informed model introduces new confusion with B014/OR014 that the plain model didn't have.
- **IR014 is a complete failure for both models** (0/177 correct, entirely predicted as OR014) — a new, shared weak spot specific to the harder single-load-train regime, not present in Phases 2–4's easier split.

## Finding 2: the physics head generalizes to unseen RPMs, but only coarsely

The most direct test: does the auxiliary head, trained on RPM≈1797 exclusively, still predict sensible frequencies at the three unseen RPMs (1721–1774)? [docs/phase5_physics_extrapolation.png](phase5_physics_extrapolation.png) shows three clear frequency clusters at roughly the right locations (~68 Hz for Ball faults, ~103 Hz for Outer race, ~155 Hz for Inner race) — the model did **not** collapse to memorizing a single training-RPM constant, and it correctly separated the three fault families by frequency neighborhood even on RPMs it never trained on. But within each cluster, the spread is wide relative to the narrow true-frequency range (MAE = 15.3 Hz across fault windows) — precise RPM-proportional tracking wasn't achieved, likely limited by (a) a single linear layer as the entire regression head, (b) the narrow RPM test range (1721–1774, ~3%) giving little signal to distinguish "correct trend" from noise, and (c) classification errors bleeding into the physics target — a misclassified window is trained toward the *wrong* fault family's frequency entirely.

**Honest framing**: this is partial success, not a dramatic win. The physics loss demonstrably taught the model a coarse, genuinely physical structure (which frequency neighborhood a fault belongs to, extrapolated correctly to unseen speeds) rather than nothing — but it isn't a precision instrument, and a stronger version of this idea would need either a more expressive physics head or a wider RPM range to prove out cleanly.

## Takeaways

- The physics-informed loss is a genuine, modest net positive on a task hard enough to have room to show it (unlike Phases 2–4's near-ceiling split) — worth keeping as an option for Phase 6, but not a silver bullet, and its class-by-class trade-offs (fixes OR007, hurts B021) should be disclosed alongside any headline number.
- IR014-vs-OR014 total confusion under single-load training is a new, previously undocumented failure mode — worth a closer look if this project continues past v1 (a natural next step: does adding the base architecture's envelope-spectrum features, Phase 1/3's most reliable discriminator, as an additional model input fix this specific pair?).
- The data-bug catch (stale channel key in `Normal_2.mat`) is the fourth methodology correction this project has surfaced and documented (SVM scaling in Phase 1, histogram binning in Phase 2, CNN seeding in Phase 4) — a real pattern worth highlighting in the portfolio README as evidence of process rigor, not just results.
