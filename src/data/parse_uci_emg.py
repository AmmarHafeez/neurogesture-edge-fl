"""Starter parser definitions for the UCI EMG Data for Gestures dataset."""

from pathlib import Path

import pandas as pd


COLUMN_NAMES = [
    "time",
    "emg_1",
    "emg_2",
    "emg_3",
    "emg_4",
    "emg_5",
    "emg_6",
    "emg_7",
    "emg_8",
    "label",
]


def read_emg_text_file(path: str | Path) -> pd.DataFrame:
    """Read one raw EMG text file into a dataframe with canonical columns."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
