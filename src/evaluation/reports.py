"""Report persistence and plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import EXPECTED_LABEL_ORDER, GESTURE_MAPPING


def save_json_report(report: dict[str, object], output_path: str | Path) -> None:
    """Write a JSON report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def save_confusion_matrix_figure(
    confusion: list[list[int]] | np.ndarray,
    output_path: str | Path,
    title: str,
    labels: list[int] | np.ndarray = EXPECTED_LABEL_ORDER,
) -> None:
    """Save a matplotlib confusion matrix figure."""
    matrix = np.asarray(confusion)
    label_order = [int(label) for label in labels]
    label_values = [
        f"{label}\n{GESTURE_MAPPING.get(label, 'unknown')}"
        for label in label_order
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(label_values)), labels=label_values, rotation=45)
    ax.set_yticks(np.arange(len(label_values)), labels=label_values)

    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            color = "white" if value > threshold else "black"
            ax.text(column, row, value, ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(report: dict[str, object], output_path: str | Path) -> None:
    """Compatibility wrapper for writing JSON reports."""
    save_json_report(report, output_path)
