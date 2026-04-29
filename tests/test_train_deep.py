import numpy as np
from pathlib import Path
import shutil
import torch
from torch import nn
from torch.utils.data import DataLoader
import uuid

from src.models.cnn1d import CNN1D
from src.preprocessing.normalization import normalize_windows_for_split
from src.training.train_deep import (
    EMGWindowDataset,
    labels_to_external,
    labels_to_zero_based,
    train_model_for_split,
    train_one_epoch,
)


def _test_workspace() -> Path:
    path = Path("tests/generated") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_cnn_forward_pass_shape() -> None:
    model = CNN1D(input_channels=8, num_classes=7)
    batch = torch.randn(4, 8, 200)

    output = model(batch)

    assert output.shape == (4, 7)


def test_label_conversion_round_trip() -> None:
    external = np.asarray([1, 2, 3, 4, 5, 6, 7])

    internal = labels_to_zero_based(external)
    restored = labels_to_external(internal)

    np.testing.assert_array_equal(internal, np.asarray([0, 1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_equal(restored, external)


def test_global_channel_zscore_uses_train_statistics_and_preserves_shape() -> None:
    X = np.arange(5 * 4 * 2, dtype=np.float32).reshape(5, 4, 2)
    train_idx = np.asarray([0, 1, 2])

    normalized, metadata = normalize_windows_for_split(
        X,
        train_idx=train_idx,
        mode="global_channel_zscore",
    )

    assert normalized.shape == X.shape
    np.testing.assert_allclose(normalized[train_idx].mean(axis=(0, 1)), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized[train_idx].std(axis=(0, 1)), 1.0, atol=1e-6)
    np.testing.assert_allclose(metadata["normalization_mean"], X[train_idx].mean(axis=(0, 1)))
    assert np.isfinite(normalized).all()


def test_per_window_channel_zscore_preserves_shape_and_is_finite() -> None:
    X = np.random.default_rng(42).normal(size=(3, 10, 2)).astype(np.float32)

    normalized, metadata = normalize_windows_for_split(
        X,
        train_idx=np.asarray([0, 1]),
        mode="per_window_channel_zscore",
    )

    assert normalized.shape == X.shape
    assert metadata["normalization_mode"] == "per_window_channel_zscore"
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-6)


def test_emg_window_dataset_returns_channel_first_tensors() -> None:
    X = np.ones((3, 200, 8), dtype=np.float32)
    y = np.asarray([1, 2, 7], dtype=np.int64)

    dataset = EMGWindowDataset(X, y)
    window, label = dataset[0]

    assert len(dataset) == 3
    assert window.shape == (8, 200)
    assert window.dtype == torch.float32
    assert label.item() == 0


def test_tiny_training_loop_runs_for_one_batch() -> None:
    X = np.random.default_rng(42).normal(size=(8, 200, 8)).astype(np.float32)
    y = np.asarray([1, 2, 3, 4, 5, 6, 7, 1], dtype=np.int64)
    dataset = EMGWindowDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = CNN1D(input_channels=8, num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert loss > 0


def test_train_model_for_split_reports_prediction_distribution() -> None:
    workspace = _test_workspace()
    try:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(14, 24, 8)).astype(np.float32)
        y = np.asarray([1, 2, 3, 4, 5, 6, 7] * 2, dtype=np.int64)
        train_idx = np.arange(7)
        test_idx = np.arange(7, 14)

        _, metrics = train_model_for_split(
            model_name="cnn1d",
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            device=torch.device("cpu"),
            epochs=1,
            batch_size=7,
            learning_rate=1e-3,
            checkpoint_path=workspace / "cnn1d_best.pt",
            normalization_mode="global_channel_zscore",
            use_class_weights=True,
            random_state=42,
        )

        assert "predicted_label_distribution" in metrics
        assert "true_label_distribution" in metrics
        assert "majority_class_baseline_accuracy" in metrics
        assert "majority_class_baseline_macro_f1" in metrics
        assert metrics["training_history"][0]["train_loss"] > 0
        assert "evaluation_macro_f1" in metrics["training_history"][0]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
