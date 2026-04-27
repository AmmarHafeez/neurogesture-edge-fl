"""Windowing helpers for EMG time-series arrays."""

import numpy as np


def sliding_windows(array: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """Create overlapping windows along the first axis of a 2D array."""
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("array must be 2D")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")
    if len(values) < window_size:
        return np.empty((0, window_size, values.shape[1]), dtype=values.dtype)

    starts = range(0, len(values) - window_size + 1, step_size)
    return np.stack([values[start : start + window_size] for start in starts])
