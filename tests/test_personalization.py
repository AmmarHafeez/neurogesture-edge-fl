from pathlib import Path
import shutil
import uuid

import numpy as np
import torch

from src.models.cnn1d import CNN1D
from src.personalization.calibrate_user import (
    configure_calibration_mode,
    sample_calibration_indices,
)
from src.personalization.evaluate_calibration import (
    evaluate_subject_calibration,
    run_personalization_experiment,
)
from src.preprocessing.normalization import normalize_windows_for_split


def _test_workspace() -> Path:
    path = Path("tests/generated") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def _synthetic_personalization_data() -> dict[str, np.ndarray]:
    X = []
    y = []
    subject_ids = []
    recording_ids = []
    gesture_names = []
    gestures = {
        1: "rest",
        2: "fist",
        3: "wrist_flexion",
        4: "wrist_extension",
        5: "radial_deviation",
        6: "ulnar_deviation",
        7: "extended_palm",
    }

    for subject in ["1", "2", "3"]:
        for label in range(1, 8):
            for repeat in range(3):
                value = label + int(subject) * 0.1 + repeat * 0.01
                X.append(np.full((24, 8), fill_value=value, dtype=np.float32))
                y.append(label)
                subject_ids.append(subject)
                recording_ids.append(f"rec_{subject}")
                gesture_names.append(gestures[label])

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "subject_ids": np.asarray(subject_ids, dtype=str),
        "recording_ids": np.asarray(recording_ids, dtype=str),
        "gesture_names": np.asarray(gesture_names, dtype=str),
        "channel_names": np.asarray([f"emg_{i}" for i in range(1, 9)], dtype=str),
        "window_size": np.asarray(24, dtype=np.int64),
        "stride": np.asarray(12, dtype=np.int64),
    }


def test_calibration_split_has_no_overlap() -> None:
    labels = np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
    subject_indices = np.arange(len(labels))

    calibration_idx, evaluation_idx, _, _ = sample_calibration_indices(
        subject_indices=subject_indices,
        labels=labels,
        calibration_per_class=1,
        random_state=42,
    )

    assert set(calibration_idx).isdisjoint(set(evaluation_idx))
    assert sorted(np.concatenate([calibration_idx, evaluation_idx]).tolist()) == list(
        subject_indices
    )


def test_calibration_sampling_respects_per_class_limit() -> None:
    labels = np.asarray([1, 1, 1, 2, 2, 3], dtype=np.int64)
    subject_indices = np.arange(len(labels))

    calibration_idx, _, calibration_counts, _ = sample_calibration_indices(
        subject_indices=subject_indices,
        labels=labels,
        calibration_per_class=2,
        random_state=42,
    )

    assert len(calibration_idx) == 5
    assert calibration_counts["1"] == 2
    assert calibration_counts["2"] == 2
    assert calibration_counts["3"] == 1


def test_last_layer_mode_freezes_feature_extractor() -> None:
    model = CNN1D(input_channels=8, num_classes=7)

    configure_calibration_mode(model, mode="last_layer")

    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_full_model_mode_leaves_all_parameters_trainable() -> None:
    model = CNN1D(input_channels=8, num_classes=7)

    configure_calibration_mode(model, mode="full_model")

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_subject_calibration_metrics_include_before_after_and_delta() -> None:
    data = _synthetic_personalization_data()
    subject_indices = np.flatnonzero(data["subject_ids"] == "3")
    train_idx = np.flatnonzero(data["subject_ids"] != "3")
    X_normalized, _ = normalize_windows_for_split(
        data["X"],
        train_idx=train_idx,
        mode="global_channel_zscore",
    )
    model = CNN1D(input_channels=8, num_classes=7)

    result = evaluate_subject_calibration(
        subject_id="3",
        subject_indices=subject_indices,
        base_model=model,
        X_normalized=X_normalized,
        y=data["y"],
        calibration_per_class=1,
        mode="last_layer",
        epochs=1,
        batch_size=8,
        learning_rate=5e-4,
        device=torch.device("cpu"),
        random_state=42,
    )

    assert result["skipped"] is False
    assert "before_personalization" in result
    assert "after_personalization" in result
    assert "delta_macro_f1" in result
    assert "delta_balanced_accuracy" in result
    assert result["calibration_size"] == 7
    assert result["evaluation_size"] == 14


def test_tiny_personalization_run_executes() -> None:
    workspace = _test_workspace()
    try:
        data = _synthetic_personalization_data()
        windows_path = workspace / "data" / "processed" / "emg_windows.npz"
        base_model_path = workspace / "models" / "cnn1d_subject_split_best.pt"
        results_path = workspace / "reports" / "metrics" / "personalization.json"
        windows_path.parent.mkdir(parents=True)
        base_model_path.parent.mkdir(parents=True)
        np.savez_compressed(windows_path, **data)

        model = CNN1D(input_channels=8, num_classes=7)
        torch.save(
            {
                "model_name": "cnn1d",
                "model_state_dict": model.state_dict(),
                "input_channels": 8,
            },
            base_model_path,
        )

        report = run_personalization_experiment(
            windows_path=windows_path,
            base_model_path=base_model_path,
            results_path=results_path,
            mode="last_layer",
            calibration_per_class=1,
            epochs=1,
            batch_size=8,
            learning_rate=5e-4,
            random_state=42,
            device_name="cpu",
        )

        assert results_path.exists()
        assert report["aggregate"]["number_of_subjects_evaluated"] >= 1
        assert "mean_delta_macro_f1" in report["aggregate"]
        assert report["normalization_mode"] == "global_channel_zscore"
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
