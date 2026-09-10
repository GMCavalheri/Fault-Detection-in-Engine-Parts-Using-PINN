"""Objective functions tuned in Phase 4: SVM (classical, cheap) and CNN
(deep learning, expensive) - both scored on validation accuracy so Optuna
and PSO can be compared on the same currency.
"""

from typing import Callable

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from src.models.cnn import Cnn1D, train_cnn


def make_svm_objective(X_train, y_train, X_val, y_val) -> Callable[[dict], float]:
    def objective(params: dict) -> float:
        model = make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=params["C"], gamma=params["gamma"])
        )
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)

    return objective


def make_cnn_objective(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    epochs: int = 12,
    device: str = "cpu",
) -> Callable[[dict], float]:
    def objective(params: dict) -> float:
        model = Cnn1D(
            num_classes=num_classes,
            width_mult=params["width_mult"],
            dropout=params["dropout"],
        )
        history = train_cnn(model, train_loader, val_loader, epochs=epochs, lr=params["lr"], device=device)
        return history[-1]["val_acc"]

    return objective
