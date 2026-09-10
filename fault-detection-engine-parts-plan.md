# Project Plan: Fault Detection in Engine Parts (Physics-Informed Deep Learning)

## Context (for the assistant continuing this project)
This is a portfolio project by Gabriel, who holds a Physics degree (UFSCar) and an MBA in AI and Big Data (USP), and recently completed a graduate monograph on **Neuroevolution applied to Physics-Informed Neural Networks (PINNs)** at USP ICMC. He wants to extend that background into industrial applications — specifically fault/anomaly detection in mechanical components — while deliberately using **different neural network architectures and optimization algorithms than genetic algorithms**, to broaden the portfolio beyond his thesis.

This project is the first in a 4-project roadmap:
1. **Fault detection in engine parts** ← this document
2. Structural/topological optimization with PINNs
3. Operator Learning (DeepONet / Fourier Neural Operator)
4. Optimizer benchmark (GA vs Bayesian Optimization vs PSO vs CMA-ES)

## Goal
Build an end-to-end pipeline that detects and diagnoses faults in mechanical components (bearings/engine parts) using sensor data (vibration signals), combining classical anomaly detection, deep learning, and — where applicable — physics-informed constraints. Compare a non-genetic optimization method against baseline tuning.

## Candidate Datasets
- **CWRU Bearing Dataset** (Case Western Reserve University) — vibration signals for bearing fault diagnosis, multiple fault types/sizes/loads. Good starting point: well-documented, widely benchmarked.
- **NASA C-MAPSS** — turbofan engine degradation simulation, used for Remaining Useful Life (RUL) prediction. Good second dataset for a predictive-maintenance angle.
- Decide which one to start with in Phase 0 (CWRU is simpler to onboard with).

## Architectures to Explore (not GA-related, deliberately diverse from the thesis)
- **1D CNN** — baseline deep learning classifier for raw/processed vibration signals
- **Autoencoder / Variational Autoencoder** — unsupervised anomaly detection via reconstruction error
- **Graph Neural Network (GNN)** — model relationships between multiple sensors/components as a graph
- **Physics-informed component (optional, stretch goal)** — incorporate known vibration/dynamics equations as a soft constraint in the loss function, connecting back to PINN expertise

## Optimization Algorithms (explicitly avoiding GA)
- **Bayesian Optimization** (via Optuna) — for hyperparameter tuning of the chosen model
- **Particle Swarm Optimization** (via pyswarms) — as a second comparison point
- Track and log results so they could later feed into Project 4 (optimizer benchmark)

## Proposed Phases

### Phase 0 — Setup
- Repo scaffolding, Python environment, dependency management
- Download and explore CWRU dataset; document signal characteristics
- Define train/val/test split strategy (careful with data leakage across operating conditions)

### Phase 1 — Classical Baseline
- Feature engineering from vibration signals (FFT, statistical features, envelope spectrum)
- Classical ML baseline (Random Forest / SVM) for fault classification
- Establish baseline metrics (accuracy, F1, confusion matrix per fault type)

### Phase 2 — Deep Learning Models
- Implement 1D CNN classifier on raw/processed signals
- Implement Autoencoder for unsupervised anomaly detection
- Compare supervised vs unsupervised approaches

### Phase 3 — Graph Neural Network Extension
- Model multi-sensor relationships as a graph (if multi-sensor data available, or simulate topology)
- Train GNN-based fault classifier/detector
- Compare against Phase 2 results

### Phase 4 — Optimization Comparison
- Tune the best-performing model from Phases 2–3 using Bayesian Optimization (Optuna)
- Repeat tuning using PSO (pyswarms)
- Compare convergence speed, final performance, and computational cost
- Document findings — this becomes a reusable case study for Project 4

### Phase 5 — Physics-Informed Extension (stretch goal)
- Identify a governing equation relevant to vibration/degradation dynamics
- Add physics-informed loss term to the best model
- Evaluate whether it improves generalization to unseen operating conditions

### Phase 6 — Productization & Portfolio Polish
- Build a Streamlit dashboard: upload a signal, get a fault prediction + confidence + anomaly score
- Wrap training/inference in a FastAPI service
- Dockerize the full stack
- Write tests (pytest) for data processing and model inference
- Write README with methodology, results, and architecture diagrams

## Suggested Tech Stack
- **Language:** Python
- **Deep Learning:** PyTorch
- **Classical ML:** scikit-learn
- **Optimization:** Optuna (Bayesian), pyswarms (PSO)
- **Signal Processing:** SciPy, NumPy
- **GNN:** PyTorch Geometric
- **Serving:** FastAPI
- **Dashboard:** Streamlit
- **Containerization:** Docker
- **Testing:** pytest
- **Repo Structure:** `/data`, `/notebooks`, `/src` (data, models, optimization, api), `/tests`, `/dashboard`, `/docker`

## Success Criteria
- Working classifier/detector with documented performance on at least 2 fault types
- Clear comparison table: classical ML vs CNN vs Autoencoder vs GNN
- Clear comparison table: Bayesian Optimization vs PSO (convergence + final metrics)
- Deployed dashboard + API, containerized
- README suitable for a public GitHub portfolio repo

## Open Decisions (to resolve at project start)
- CWRU only, or CWRU + C-MAPSS combined scope? **Resolved: both** — CWRU drove Phases 1–6; C-MAPSS RUL regression (Gradient Boosting + LSTM, FD001) was addressed as a post-Phase-6 addendum, see [docs/cmapss_results.md](docs/cmapss_results.md).
- Repo name and GitHub Pages/dashboard hosting approach
- Whether Phase 5 (physics-informed extension) is in scope for v1 or deferred to a v2 — **Resolved: in scope**, see [docs/phase5_results.md](docs/phase5_results.md).
