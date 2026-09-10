"""Shared scaffolding for comparing Bayesian Optimization (Optuna) vs. PSO
(pyswarms) as hyperparameter tuners - Phase 4 of the project plan.

Both optimizers tune the exact same continuous search space and call the
exact same `objective_fn(params) -> score` (higher is better), wrapped in
`TrackedObjective` so every individual model fit/train call - regardless of
which library triggered it - is logged with its wall-clock time. That's what
makes the two runs comparable on convergence speed and computational cost,
not just final score.

Search space format: {"name": ("float", low, high)} or
{"name": ("float_log", low, high)} for parameters best searched in log10
space (e.g. SVM's C/gamma span orders of magnitude).
"""

import time
from typing import Callable

import numpy as np
import optuna
import pyswarms as ps

ParamSpace = dict[str, tuple[str, float, float]]


class TrackedObjective:
    """Wraps an objective so every call is logged with wall-clock time,
    regardless of which optimizer (Optuna or PSO) triggered it.
    """

    def __init__(self, objective_fn: Callable[[dict], float]):
        self.objective_fn = objective_fn
        self.history: list[dict] = []
        self._start = time.perf_counter()

    def __call__(self, params: dict) -> float:
        score = self.objective_fn(params)
        self.history.append(
            {
                "eval": len(self.history),
                "wall_time_s": time.perf_counter() - self._start,
                "params": dict(params),
                "score": score,
            }
        )
        return score


def _best_so_far(history: list[dict]) -> list[float]:
    best, out = -np.inf, []
    for row in history:
        best = max(best, row["score"])
        out.append(best)
    return out


def run_optuna(
    objective_fn: Callable[[dict], float], param_space: ParamSpace, n_trials: int, seed: int = 42
) -> list[dict]:
    tracked = TrackedObjective(objective_fn)

    def optuna_objective(trial: optuna.Trial) -> float:
        params = {}
        for name, (kind, low, high) in param_space.items():
            if kind == "float_log":
                params[name] = trial.suggest_float(name, low, high, log=True)
            else:
                params[name] = trial.suggest_float(name, low, high)
        return tracked(params)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(optuna_objective, n_trials=n_trials)

    history = tracked.history
    best_so_far = _best_so_far(history)
    for row, best in zip(history, best_so_far):
        row["best_so_far"] = best
    return history


def _decode_particle(x_row: np.ndarray, param_space: ParamSpace) -> dict:
    params = {}
    for i, (name, (kind, _, _)) in enumerate(param_space.items()):
        value = x_row[i]
        params[name] = 10**value if kind == "float_log" else value
    return params


def run_pso(
    objective_fn: Callable[[dict], float],
    param_space: ParamSpace,
    n_particles: int,
    n_iters: int,
    seed: int = 42,
) -> list[dict]:
    tracked = TrackedObjective(objective_fn)

    lower, upper = [], []
    for kind, low, high in param_space.values():
        bound = (np.log10(low), np.log10(high)) if kind == "float_log" else (low, high)
        lower.append(bound[0])
        upper.append(bound[1])

    def pso_cost(x: np.ndarray) -> np.ndarray:
        # pyswarms passes (n_particles, n_dims) and expects a cost to MINIMIZE
        return np.array([-tracked(_decode_particle(row, param_space)) for row in x])

    np.random.seed(seed)
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(param_space),
        options={"c1": 0.5, "c2": 0.3, "w": 0.9},
        bounds=(np.array(lower), np.array(upper)),
    )
    optimizer.optimize(pso_cost, iters=n_iters, verbose=False)

    history = tracked.history
    best_so_far = _best_so_far(history)
    for row, best in zip(history, best_so_far):
        row["best_so_far"] = best
    return history
