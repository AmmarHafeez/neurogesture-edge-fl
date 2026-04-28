"""Train/test split helpers for EMG window evaluation."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.model_selection import train_test_split


LOGGER = logging.getLogger(__name__)


def make_random_split(
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a random train/test split, stratified when feasible."""
    labels = np.asarray(y)
    indices = np.arange(len(labels))
    if len(indices) < 2:
        raise ValueError("At least two samples are required for a train/test split")

    unique_labels, counts = np.unique(labels, return_counts=True)
    test_count = int(np.ceil(len(labels) * test_size))
    train_count = len(labels) - test_count
    can_stratify = (
        len(unique_labels) > 1
        and counts.min() >= 2
        and test_count >= len(unique_labels)
        and train_count >= len(unique_labels)
    )

    if not can_stratify:
        LOGGER.warning("Falling back to an unstratified random split")

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if can_stratify else None,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def make_subject_split(
    subject_ids: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create a deterministic split that holds out complete subjects."""
    subjects = np.asarray(subject_ids).astype(str)
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        raise ValueError("At least two subjects are required for a subject split")

    rng = np.random.default_rng(random_state)
    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    n_test_subjects = max(1, int(np.ceil(len(unique_subjects) * test_size)))
    n_test_subjects = min(n_test_subjects, len(unique_subjects) - 1)
    test_subjects = sorted(shuffled_subjects[:n_test_subjects].tolist())

    test_mask = np.isin(subjects, test_subjects)
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Subject split produced an empty train or test set")

    return train_idx, test_idx, test_subjects
