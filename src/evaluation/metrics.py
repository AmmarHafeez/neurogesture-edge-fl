"""Classification metric helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


EXPECTED_LABEL_ORDER = [1, 2, 3, 4, 5, 6, 7]
GESTURE_MAPPING = {
    1: "rest",
    2: "fist",
    3: "wrist_flexion",
    4: "wrist_extension",
    5: "radial_deviation",
    6: "ulnar_deviation",
    7: "extended_palm",
}
LOW_TEST_SAMPLE_THRESHOLD = 50


def label_distribution(
    labels: np.ndarray,
    label_order: list[int] | np.ndarray = EXPECTED_LABEL_ORDER,
) -> dict[str, int]:
    """Count labels using a fixed label order."""
    values = np.asarray(labels).astype(int)
    return {
        str(label): int(np.sum(values == label))
        for label in np.asarray(label_order).astype(int)
    }


def gesture_distribution(
    labels: np.ndarray,
    label_order: list[int] | np.ndarray = EXPECTED_LABEL_ORDER,
) -> dict[str, int]:
    """Count gestures using the fixed gesture mapping."""
    counts = label_distribution(labels, label_order=label_order)
    return {
        GESTURE_MAPPING[int(label)]: count
        for label, count in counts.items()
    }


def missing_labels(
    labels: np.ndarray,
    label_order: list[int] | np.ndarray = EXPECTED_LABEL_ORDER,
) -> list[int]:
    """Return expected labels not present in an array."""
    counts = label_distribution(labels, label_order=label_order)
    return [int(label) for label, count in counts.items() if count == 0]


def low_test_sample_warnings(
    y_true: np.ndarray,
    label_order: list[int] | np.ndarray = EXPECTED_LABEL_ORDER,
    threshold: int = LOW_TEST_SAMPLE_THRESHOLD,
) -> list[str]:
    """Create warnings for expected labels with low test support."""
    counts = label_distribution(y_true, label_order=label_order)
    warnings = []
    for label, count in counts.items():
        label_id = int(label)
        if count < threshold:
            gesture = GESTURE_MAPPING[label_id]
            warnings.append(
                f"label {label_id} ({gesture}) has {count} test samples "
                f"(<{threshold})"
            )
    return warnings


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    train_size: int | None = None,
    test_subjects: list[str] | None = None,
) -> dict[str, object]:
    """Compute baseline classification metrics as JSON-serializable values."""
    true = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    _ = labels
    labels_array = np.asarray(EXPECTED_LABEL_ORDER, dtype=int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=labels_array,
        zero_division=0,
    )
    train_labels = np.asarray(y_train).astype(int) if y_train is not None else None
    test_distribution = label_distribution(true, label_order=labels_array)
    predicted_not_in_test = sorted(
        int(label) for label in set(pred.tolist()) - set(true.tolist())
    )

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(
            f1_score(
                true,
                pred,
                labels=labels_array,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true,
                pred,
                labels=labels_array,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(np.mean(recall)),
        "per_class_precision": {
            str(label): float(value) for label, value in zip(labels_array, precision)
        },
        "per_class_recall": {
            str(label): float(value) for label, value in zip(labels_array, recall)
        },
        "per_class_f1": {
            str(label): float(value) for label, value in zip(labels_array, f1)
        },
        "confusion_matrix": confusion_matrix(true, pred, labels=labels_array).tolist(),
        "test_size": int(len(true)),
        "test_label_distribution": test_distribution,
        "test_gesture_distribution": gesture_distribution(
            true,
            label_order=labels_array,
        ),
        "labels_missing_from_test": missing_labels(true, label_order=labels_array),
        "labels_predicted_not_in_test": predicted_not_in_test,
        "class_imbalance_warnings": low_test_sample_warnings(
            true,
            label_order=labels_array,
        ),
    }
    if train_labels is not None:
        metrics["train_size"] = int(len(train_labels))
        metrics["train_label_distribution"] = label_distribution(
            train_labels,
            label_order=labels_array,
        )
        metrics["train_gesture_distribution"] = gesture_distribution(
            train_labels,
            label_order=labels_array,
        )
        metrics["labels_missing_from_train"] = missing_labels(
            train_labels,
            label_order=labels_array,
        )
    elif train_size is not None:
        metrics["train_size"] = int(train_size)
    if test_subjects is not None:
        metrics["test_subjects"] = [str(subject) for subject in test_subjects]
    return metrics
