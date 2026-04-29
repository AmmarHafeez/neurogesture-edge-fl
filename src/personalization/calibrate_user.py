"""User calibration helpers for EMG personalization experiments."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import EXPECTED_LABEL_ORDER, label_distribution
from src.models.cnn1d import CNN1D
from src.training.train_deep import (
    EMGWindowDataset,
    compute_class_weights,
    make_dataloader,
    train_one_epoch,
)


LOGGER = logging.getLogger(__name__)


def sample_calibration_indices(
    subject_indices: np.ndarray,
    labels: np.ndarray,
    calibration_per_class: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    """Sample calibration windows per class and return remaining evaluation windows."""
    if calibration_per_class <= 0:
        raise ValueError("calibration_per_class must be positive")

    indices = np.asarray(subject_indices, dtype=int)
    y = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(random_state)

    calibration_parts: list[np.ndarray] = []
    for label in EXPECTED_LABEL_ORDER:
        label_indices = indices[y[indices] == label]
        if len(label_indices) == 0:
            continue
        shuffled = label_indices.copy()
        rng.shuffle(shuffled)
        calibration_parts.append(shuffled[:calibration_per_class])

    if calibration_parts:
        calibration_idx = np.sort(np.concatenate(calibration_parts))
    else:
        calibration_idx = np.empty((0,), dtype=int)

    calibration_set = set(calibration_idx.tolist())
    evaluation_idx = np.asarray(
        [index for index in indices.tolist() if index not in calibration_set],
        dtype=int,
    )

    calibration_counts = label_distribution(y[calibration_idx])
    evaluation_counts = label_distribution(y[evaluation_idx])
    return calibration_idx, evaluation_idx, calibration_counts, evaluation_counts


def build_cnn_from_checkpoint(
    checkpoint_path: str | Path,
    input_channels: int = 8,
    device: torch.device | None = None,
) -> CNN1D:
    """Load a CNN-1D model checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Base CNN checkpoint does not exist: {path}. "
            "Run train_deep.py first to create models/cnn1d_subject_split_best.pt."
        )

    target_device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=target_device, weights_only=False)
    model_input_channels = int(checkpoint.get("input_channels", input_channels))
    model = CNN1D(input_channels=model_input_channels, num_classes=len(EXPECTED_LABEL_ORDER))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    return model


def configure_calibration_mode(model: nn.Module, mode: str = "last_layer") -> None:
    """Set trainable parameters for a personalization mode."""
    if mode == "last_layer":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if not hasattr(model, "classifier"):
            raise ValueError("last_layer mode requires a model.classifier module")
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if mode == "full_model":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Unsupported calibration mode: {mode}")


def fine_tune_model(
    base_model: nn.Module,
    X_normalized: np.ndarray,
    y: np.ndarray,
    calibration_idx: np.ndarray,
    mode: str = "last_layer",
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    device: torch.device | None = None,
    random_state: int = 42,
) -> nn.Module:
    """Fine-tune a copy of a global model on calibration windows."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if len(calibration_idx) == 0:
        raise ValueError("calibration_idx must contain at least one window")

    torch.manual_seed(random_state)
    target_device = device or torch.device("cpu")
    model = copy.deepcopy(base_model).to(target_device)
    configure_calibration_mode(model, mode=mode)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("Calibration mode left no trainable parameters")

    dataset = EMGWindowDataset(X_normalized, y)
    dataloader = make_dataloader(
        dataset,
        np.asarray(calibration_idx, dtype=int),
        batch_size=batch_size,
        shuffle=True,
    )
    class_weights = compute_class_weights(y[calibration_idx]).to(target_device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=1e-4,
    )

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=target_device,
            max_grad_norm=5.0,
        )
        LOGGER.info("Personalization epoch %s loss=%.6f", epoch, loss)

    return model


def make_eval_loader(
    X_normalized: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> DataLoader:
    """Create a deterministic evaluation loader for personalization."""
    dataset = EMGWindowDataset(X_normalized, y)
    return make_dataloader(
        dataset,
        np.asarray(indices, dtype=int),
        batch_size=batch_size,
        shuffle=False,
    )
