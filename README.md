# Fault Detection in Engine Parts (Physics-Informed Deep Learning)

Fault/anomaly detection and diagnosis for mechanical components (bearings and turbofan engine parts) using sensor data, combining classical anomaly detection, deep learning, and physics-informed constraints. Built with architectures and optimization algorithms deliberately distinct from genetic algorithms, to broaden a portfolio started with a thesis on Neuroevolution applied to Physics-Informed Neural Networks (PINNs).

This is **project 1 of a 4-project roadmap**:
1. **Fault detection in engine parts** ← this repo
2. Structural/topological optimization with PINNs
3. Operator Learning (DeepONet / Fourier Neural Operator)
4. Optimizer benchmark (GA vs Bayesian Optimization vs PSO vs CMA-ES)

## Status

**Phase 0 — Setup** (in progress). See [fault-detection-engine-parts-plan.md](fault-detection-engine-parts-plan.md) for the full phase-by-phase project plan.

## Datasets

- **CWRU Bearing Dataset** — vibration signals for bearing fault diagnosis (multiple fault types/sizes/loads).
- **NASA C-MAPSS** — turbofan engine degradation simulation, for Remaining Useful Life (RUL) prediction.

Dataset details (sampling rates, labels, split strategy) are documented in [docs/datasets.md](docs/datasets.md).

## Repo Structure

```
data/            raw and processed datasets (not versioned, see .gitignore)
notebooks/       exploration and analysis notebooks
src/data/        data loading and preprocessing
src/models/      classical ML, CNN, autoencoder, GNN model implementations
src/optimization/  Bayesian optimization (Optuna) and PSO (pyswarms) tuning
src/physics/     physics-informed loss terms / PINN components
src/api/         FastAPI inference service
tests/           pytest test suite
dashboard/       Streamlit dashboard
docker/          containerization
docs/            project documentation
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

> **Notes on this environment (Windows):**
> - The project targets Python 3.11+, but `uv`'s managed-Python downloader hit an environment-specific bug on the original dev machine (junction/symlink creation failed after extracting the interpreter — likely AppData\Roaming sync interference). As a workaround, the `.venv` currently runs on the system's Anaconda Python 3.9.7 (`requires-python = ">=3.9"` in `pyproject.toml`). Switch back to a `uv`-managed 3.11 interpreter when convenient: `uv python install 3.11`, then `uv sync --python 3.11` and bump `requires-python`.
> - Because the venv is built on Anaconda's Python, running the Jupyter kernel (e.g. `jupyter nbconvert --execute`) needs `sqlite3.dll` from `anaconda3\Library\bin` on `PATH` (normally added by `conda activate`, which we're bypassing). Add it manually if you hit `DLL load failed while importing _sqlite3`: `$env:Path = "C:\Users\lcava\anaconda3\Library\bin;$env:Path"`.

Later phases need optional dependency groups (kept out of the base install to stay light):

```bash
uv sync --extra dl          # PyTorch (Phase 2)
uv sync --extra gnn         # PyTorch Geometric (Phase 3)
uv sync --extra opt         # Optuna, pyswarms (Phase 4)
uv sync --extra serve       # FastAPI (Phase 6)
uv sync --extra dashboard   # Streamlit (Phase 6)
```

Run the data exploration notebook:

```bash
uv run jupyter notebook notebooks/00_data_exploration.ipynb
```

Run tests:

```bash
uv run pytest
```
