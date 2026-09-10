# Datasets

## CWRU Bearing Dataset

Source: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file), 12k Drive End files.

Downloaded into `data/raw/cwru/` (40 files, ~115 MB): `Normal_{0..3}.mat` (baseline) plus all 4 loads × 3 fault types (Inner Race `IR`, Ball `B`, Outer Race centered-at-6:00 `OR`) × 3 fault diameters (0.007", 0.014", 0.021") — full cross-condition coverage as of Phase 2, expanded from the load-0-only subset downloaded in Phase 0.

- **Sampling rate**: 12,000 samples/second
- **Format**: MATLAB `.mat`, one struct-like dict per file
- **Channels** (variable name suffixes, not all present in every file):
  - `_DE_time` — drive-end accelerometer (always present)
  - `_FE_time` — fan-end accelerometer (always present)
  - `_BA_time` — base accelerometer (present on fault files, absent on `Normal_*`)
  - `RPM` — motor rotational speed, scalar
- **Signal length**: varies per file — `Normal_0` has 243,938 samples/channel; `IR007_0` has 121,265. Fixed-length windows must be extracted per file, not assumed uniform.
- **Fault labels present**: Normal, Inner Race (IR), Ball (B), Outer Race centered (OR@6) — at fault diameters 0.007"/0.014"/0.021"
- **Operating conditions**: motor load 0–3 HP (approx. 1797/1772/1750/1730 rpm) — all 10 classes (Normal + 9 fault-type/diameter combos) now have all 4 loads, supporting held-out-load generalization testing across the full class set (used from Phase 2 onward).

## NASA C-MAPSS

Source: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), "Turbofan Engine Degradation Simulation" data set (item 6).

Downloaded and extracted into `data/raw/cmapss/`: `train_FD001..FD004.txt`, `test_FD001..FD004.txt`, `RUL_FD001..FD004.txt`, plus `readme.txt` and the damage-propagation-modeling reference PDF.

- **Format**: whitespace-delimited text, no header. 26 columns: `unit_id`, `cycle`, 3 operational settings, 21 sensor measurements.
- **Subsets**:
  | Subset | Train trajectories | Test trajectories | Operating conditions | Fault modes |
  |---|---|---|---|---|
  | FD001 | 100 | 100 | 1 (sea level) | 1 (HPC degradation) |
  | FD002 | 260 | 259 | 6 | 1 (HPC degradation) |
  | FD003 | 100 | 100 | 1 (sea level) | 2 (HPC + Fan degradation) |
  | FD004 | 248 | 249 | 6 | 2 (HPC + Fan degradation) |
- **FD001 verified shape**: `train_FD001.txt` is 20,631 rows × 26 columns across 100 engine units; per-unit trajectory length (cycles to failure) varies (e.g. unit 1 = 192 cycles, unit 2 = 287 cycles).
- **Target**: Remaining Useful Life (RUL) — train files run each unit to failure; test files are truncated before failure with true RUL given separately in `RUL_FD00X.txt`.
- **Starting subset**: FD001 (single condition, single fault mode) — simplest onboarding. FD002–FD004 available for later generalization work.

## Train / Val / Test Split Strategy

Splits are made **by unit/operating condition, not by randomly shuffling individual samples**, to avoid leakage between train and test sets:

- **CWRU**: split by bearing fault instance (each combination of fault type/size/load) rather than by random windows within a signal. Windows drawn from the same continuous recording must not appear in both train and test — otherwise the model can memorize a specific recording's noise characteristics rather than learning the fault signature. With full 4-load coverage across all classes (since Phase 2), the standard split is by held-out load: train on loads 0–2, test on load 3 — a genuine cross-condition generalization test, not just a within-recording holdout.
- **C-MAPSS**: split by engine unit ID — each unit's full run-to-failure trajectory belongs entirely to one split. This matches the dataset's own design: it already ships with separate train/test files per subset, with the test set truncated before failure and true RUL given in the companion `RUL_FD00X.txt` file. No custom splitting needed for the baseline evaluation protocol; any additional internal train/val split should still respect unit boundaries.

This matches the leakage-awareness note in the project plan's Phase 0.
