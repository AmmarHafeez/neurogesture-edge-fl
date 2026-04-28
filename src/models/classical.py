"""Classical baseline model factories and feature preparation."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing.features import mean_absolute_value


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """Flatten EMG windows from 3D tensors to 2D feature matrices."""
    values = np.asarray(X)
    if values.ndim != 3:
        raise ValueError("X must have shape (n_windows, window_size, n_channels)")
    return values.reshape(values.shape[0], values.shape[1] * values.shape[2])


def extract_window_features(X: np.ndarray, mode: str = "flatten") -> np.ndarray:
    """Prepare classical-model features from EMG windows."""
    if mode == "flatten":
        return flatten_windows(X)
    if mode == "mean_absolute_value":
        values = np.asarray(X)
        if values.ndim != 3:
            raise ValueError("X must have shape (n_windows, window_size, n_channels)")
        return np.asarray([mean_absolute_value(window) for window in values])
    raise ValueError(f"Unsupported feature mode: {mode}")


def build_logistic_regression(random_state: int = 42) -> Pipeline:
    """Build a balanced logistic regression baseline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Build a balanced random forest baseline."""
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )


def build_classical_models(random_state: int = 42) -> dict[str, object]:
    """Build all configured classical baseline models."""
    return {
        "logistic_regression": build_logistic_regression(random_state=random_state),
        "random_forest": build_random_forest(random_state=random_state),
    }


def build_baseline_model(model_name: str = "random_forest", random_state: int = 42):
    """Build one named classical baseline model."""
    models = build_classical_models(random_state=random_state)
    if model_name not in models:
        raise ValueError(f"Unknown baseline model: {model_name}")
    return models[model_name]
