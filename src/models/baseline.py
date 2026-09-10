"""Classical ML baseline for CWRU fault classification.

Train/test splitting follows docs/datasets.md: windows from the same source
file must not be split across train and test. Two split strategies are
provided:
  - `time_contiguous_split`: for the common case of one file per class, holds
    out the last `test_fraction` of each file's windows in time order (so
    train and test never come from overlapping instants).
  - `held_out_files_split`: for cross-condition generalization checks, holds
    out entire files (e.g. one load condition) for testing.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def time_contiguous_split(
    file_ids: np.ndarray, test_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Per file, the last `test_fraction` of windows (in original/time order)
    go to test, the rest to train. Returns boolean (train_mask, test_mask).
    """
    train_mask = np.ones(len(file_ids), dtype=bool)
    test_mask = np.zeros(len(file_ids), dtype=bool)
    for file_id in np.unique(file_ids):
        idx = np.where(file_ids == file_id)[0]
        n_test = max(1, int(round(len(idx) * test_fraction)))
        test_idx = idx[-n_test:]
        train_mask[test_idx] = False
        test_mask[test_idx] = True
    return train_mask, test_mask


def held_out_files_split(
    file_ids: np.ndarray, test_file_ids: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Whole files go entirely to train or entirely to test."""
    test_mask = np.isin(file_ids, list(test_file_ids))
    return ~test_mask, test_mask


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, random_state: int = 42):
    # RBF-kernel SVM is sensitive to feature scale (our features span orders of
    # magnitude, e.g. fd_energy ~1e5 vs td_mean ~1e-2), so scale first.
    model = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", random_state=random_state)
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    labels = sorted(set(y_test) | set(y_pred))
    return {
        "report": classification_report(y_test, y_pred, labels=labels, zero_division=0),
        "report_dict": classification_report(
            y_test, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
        "y_pred": y_pred,
    }
