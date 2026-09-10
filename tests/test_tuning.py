import numpy as np

from src.optimization.tuning import TrackedObjective, run_optuna, run_pso


def _quadratic_objective(params: dict) -> float:
    # peak at x=2, y=3, score=0 (maximize)
    return -((params["x"] - 2) ** 2) - (params["y"] - 3) ** 2


def test_tracked_objective_logs_every_call():
    tracked = TrackedObjective(_quadratic_objective)
    tracked({"x": 0, "y": 0})
    tracked({"x": 1, "y": 1})

    assert len(tracked.history) == 2
    assert tracked.history[0]["eval"] == 0
    assert tracked.history[1]["eval"] == 1
    assert tracked.history[1]["wall_time_s"] >= tracked.history[0]["wall_time_s"]
    assert tracked.history[0]["score"] == _quadratic_objective({"x": 0, "y": 0})


def test_run_optuna_improves_over_time():
    param_space = {"x": ("float", -5.0, 5.0), "y": ("float", -5.0, 5.0)}
    history = run_optuna(_quadratic_objective, param_space, n_trials=20, seed=0)

    assert len(history) == 20
    best_so_far = [row["best_so_far"] for row in history]
    assert best_so_far == sorted(best_so_far)  # non-decreasing
    assert best_so_far[-1] > best_so_far[0]  # optimization actually improved
    assert best_so_far[-1] > -5.0  # found something reasonably close to the peak (0)


def test_run_pso_improves_over_time():
    param_space = {"x": ("float", -5.0, 5.0), "y": ("float", -5.0, 5.0)}
    history = run_pso(_quadratic_objective, param_space, n_particles=6, n_iters=5, seed=0)

    assert len(history) == 30  # 6 particles x 5 iterations
    best_so_far = [row["best_so_far"] for row in history]
    assert best_so_far == sorted(best_so_far)
    assert best_so_far[-1] > best_so_far[0]
    assert best_so_far[-1] > -5.0


def test_log_space_param_decoded_correctly_by_pso():
    seen_params = []

    def objective(params: dict) -> float:
        seen_params.append(params["C"])
        return -abs(np.log10(params["C"]) - 1)  # peak at C=10

    param_space = {"C": ("float_log", 1e-2, 1e3)}
    run_pso(objective, param_space, n_particles=4, n_iters=2, seed=0)

    # every decoded C must land inside the requested (non-log) bounds
    assert all(1e-2 <= c <= 1e3 for c in seen_params)
