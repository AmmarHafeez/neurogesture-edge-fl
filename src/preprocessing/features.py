"""Starter feature extraction helpers."""

import numpy as np


def mean_absolute_value(window: np.ndarray) -> np.ndarray:
    """Compute mean absolute value per channel for one EMG window."""
    values = np.asarray(window)
    if values.ndim != 2:
        raise ValueError("window must be 2D")
    return np.mean(np.abs(values), axis=0)
