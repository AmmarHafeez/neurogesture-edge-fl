"""Dataset download helper placeholder.

The UCI dataset should be placed manually under data/raw for the first skeleton.
"""

from pathlib import Path


DEFAULT_RAW_DIR = Path("data/raw/EMG_data_for_gestures-master")


def dataset_expected_path() -> Path:
    """Return the expected local raw dataset path."""
    return DEFAULT_RAW_DIR
