"""Windowing helpers for EMG time-series arrays and parsed EMG samples."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


EMG_CHANNEL_COLUMNS = [
    "emg_1",
    "emg_2",
    "emg_3",
    "emg_4",
    "emg_5",
    "emg_6",
    "emg_7",
    "emg_8",
]

WINDOW_SOURCE_COLUMNS = [
    "subject_id",
    "recording_id",
    "time",
    *EMG_CHANNEL_COLUMNS,
    "label_id",
    "gesture",
]


@dataclass(frozen=True)
class WindowedDataset:
    """In-memory representation of fixed-length EMG windows."""

    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    recording_ids: np.ndarray
    gesture_names: np.ndarray
    channel_names: np.ndarray
    window_size: int
    stride: int


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


def validate_window_source_columns(dataframe: pd.DataFrame) -> None:
    """Validate that parsed EMG samples contain the columns needed for windowing."""
    missing_columns = [
        column for column in WINDOW_SOURCE_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required windowing columns: {missing_columns}")


def _empty_windowed_dataset(window_size: int, stride: int) -> WindowedDataset:
    """Create an empty window dataset with stable shapes and dtypes."""
    return WindowedDataset(
        X=np.empty((0, window_size, len(EMG_CHANNEL_COLUMNS)), dtype=np.float32),
        y=np.empty((0,), dtype=np.int64),
        subject_ids=np.empty((0,), dtype=str),
        recording_ids=np.empty((0,), dtype=str),
        gesture_names=np.empty((0,), dtype=str),
        channel_names=np.asarray(EMG_CHANNEL_COLUMNS, dtype=str),
        window_size=window_size,
        stride=stride,
    )


def build_windows_from_dataframe(
    dataframe: pd.DataFrame,
    window_size: int = 200,
    stride: int = 100,
) -> WindowedDataset:
    """Build fixed-length EMG windows from parsed sample rows.

    Rows are grouped by subject, recording, and gesture so windows do not cross
    recording boundaries or mix gesture classes in this first implementation.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    validate_window_source_columns(dataframe)

    window_arrays: list[np.ndarray] = []
    labels: list[int] = []
    subject_ids: list[str] = []
    recording_ids: list[str] = []
    gesture_names: list[str] = []

    group_columns = ["subject_id", "recording_id", "gesture"]
    for (subject_id, recording_id, gesture), group in dataframe.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        sorted_group = group.sort_values("time")
        label_ids = sorted_group["label_id"].dropna().unique()
        if len(label_ids) != 1:
            continue

        values = sorted_group[EMG_CHANNEL_COLUMNS].to_numpy(dtype=np.float32)
        windows = sliding_windows(values, window_size=window_size, step_size=stride)
        if len(windows) == 0:
            continue

        window_arrays.extend(windows.astype(np.float32, copy=False))
        labels.extend([int(label_ids[0])] * len(windows))
        subject_ids.extend([str(subject_id)] * len(windows))
        recording_ids.extend([str(recording_id)] * len(windows))
        gesture_names.extend([str(gesture)] * len(windows))

    if not window_arrays:
        return _empty_windowed_dataset(window_size=window_size, stride=stride)

    return WindowedDataset(
        X=np.stack(window_arrays).astype(np.float32, copy=False),
        y=np.asarray(labels, dtype=np.int64),
        subject_ids=np.asarray(subject_ids, dtype=str),
        recording_ids=np.asarray(recording_ids, dtype=str),
        gesture_names=np.asarray(gesture_names, dtype=str),
        channel_names=np.asarray(EMG_CHANNEL_COLUMNS, dtype=str),
        window_size=window_size,
        stride=stride,
    )
