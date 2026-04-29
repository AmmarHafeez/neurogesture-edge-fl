"""Normalization helpers for EMG features and windows."""

from __future__ import annotations

import numpy as np


def zscore(array: np.ndarray, axis: int = 0, epsilon: float = 1e-8) -> np.ndarray:
    """Apply z-score normalization with a small numerical guard."""
    values = np.asarray(array, dtype=float)
    mean = values.mean(axis=axis, keepdims=True)
    std = values.std(axis=axis, keepdims=True)
    return (values - mean) / (std + epsilon)


def compute_global_channel_stats(
    X_train: np.ndarray,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from training windows only."""
    values = np.asarray(X_train, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("X_train must have shape (n_windows, window_size, n_channels)")
    mean = values.mean(axis=(0, 1))
    std = values.std(axis=(0, 1))
    return mean.astype(np.float32), np.maximum(std, epsilon).astype(np.float32)


def apply_global_channel_zscore(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply per-channel z-score normalization to EMG windows."""
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("X must have shape (n_windows, window_size, n_channels)")
    return ((values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(
        np.float32,
        copy=False,
    )


def apply_per_window_channel_zscore(
    X: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Apply z-score normalization per window and channel over time."""
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("X must have shape (n_windows, window_size, n_channels)")
    mean = values.mean(axis=1, keepdims=True)
    std = np.maximum(values.std(axis=1, keepdims=True), epsilon)
    return ((values - mean) / std).astype(np.float32, copy=False)


def normalize_windows_for_split(
    X: np.ndarray,
    train_idx: np.ndarray,
    mode: str = "global_channel_zscore",
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Normalize all windows using train-split statistics where applicable."""
    values = np.asarray(X, dtype=np.float32)
    indices = np.asarray(train_idx, dtype=int)
    if len(indices) == 0:
        raise ValueError("train_idx must contain at least one index")

    if mode == "global_channel_zscore":
        mean, std = compute_global_channel_stats(values[indices], epsilon=epsilon)
        normalized = apply_global_channel_zscore(values, mean=mean, std=std)
        metadata = {
            "normalization_mode": mode,
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
        }
        return normalized, metadata

    if mode == "per_window_channel_zscore":
        normalized = apply_per_window_channel_zscore(values, epsilon=epsilon)
        train_window_means = values[indices].mean(axis=1)
        train_window_stds = np.maximum(values[indices].std(axis=1), epsilon)
        metadata = {
            "normalization_mode": mode,
            "normalization_mean": train_window_means.mean(axis=0).astype(float).tolist(),
            "normalization_std": train_window_stds.mean(axis=0).astype(float).tolist(),
        }
        return normalized, metadata

    raise ValueError(f"Unsupported normalization mode: {mode}")
