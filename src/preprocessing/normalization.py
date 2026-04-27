"""Normalization helpers for EMG features and windows."""

import numpy as np


def zscore(array: np.ndarray, axis: int = 0, epsilon: float = 1e-8) -> np.ndarray:
    """Apply z-score normalization with a small numerical guard."""
    values = np.asarray(array, dtype=float)
    mean = values.mean(axis=axis, keepdims=True)
    std = values.std(axis=axis, keepdims=True)
    return (values - mean) / (std + epsilon)
