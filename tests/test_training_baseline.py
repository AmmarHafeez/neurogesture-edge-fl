import json
from pathlib import Path
import shutil
import uuid
import warnings

import numpy as np

from src.evaluation.metrics import classification_metrics
from src.evaluation.splits import make_random_split, make_subject_split
from src.models.classical import flatten_windows
from src.training.train_baseline import run_training


def _test_workspace() -> Path:
    path = Path("tests/generated") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def _synthetic_window_dataset() -> dict[str, np.ndarray]:
    X = []
    y = []
    subject_ids = []
    recording_ids = []
    gesture_names = []
    label_to_gesture = {1: "rest", 2: "fist", 3: "wrist_flexion"}

    for subject in range(6):
        for label in [1, 2, 3]:
            for repeat in range(2):
                X.append(np.full((4, 2), fill_value=label + repeat * 0.01))
                y.append(label)
                subject_ids.append(str(subject))
                recording_ids.append(f"rec_{subject}")
                gesture_names.append(label_to_gesture[label])

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "subject_ids": np.asarray(subject_ids, dtype=str),
        "recording_ids": np.asarray(recording_ids, dtype=str),
        "gesture_names": np.asarray(gesture_names, dtype=str),
        "channel_names": np.asarray(["emg_1", "emg_2"], dtype=str),
        "window_size": np.asarray(4, dtype=np.int64),
        "stride": np.asarray(2, dtype=np.int64),
    }


def test_random_split_preserves_expected_sizes() -> None:
    y = np.asarray([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    train_idx, test_idx = make_random_split(y, test_size=0.3, random_state=42)

    assert len(train_idx) == 7
    assert len(test_idx) == 3
    assert set(train_idx).isdisjoint(set(test_idx))


def test_subject_split_has_no_subject_leakage() -> None:
    subject_ids = np.asarray(["1", "1", "2", "2", "3", "3", "4", "4"])

    train_idx, test_idx, test_subjects = make_subject_split(
        subject_ids,
        test_size=0.25,
        random_state=42,
    )

    train_subjects = set(subject_ids[train_idx])
    held_out_subjects = set(subject_ids[test_idx])
    assert train_subjects.isdisjoint(held_out_subjects)
    assert held_out_subjects == set(test_subjects)


def test_flatten_windows_converts_3d_to_2d() -> None:
    X = np.arange(24).reshape(3, 4, 2)

    flattened = flatten_windows(X)

    assert flattened.shape == (3, 8)
    np.testing.assert_array_equal(flattened[0], np.arange(8))


def test_metric_report_contains_required_keys() -> None:
    metrics = classification_metrics(
        y_true=np.asarray([1, 1, 2, 2]),
        y_pred=np.asarray([1, 2, 2, 2]),
        y_train=np.asarray([1, 1, 2, 2, 3, 3]),
        train_size=6,
    )

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "balanced_accuracy" in metrics
    assert metrics["train_size"] == 6
    assert metrics["test_size"] == 4
    assert list(metrics["per_class_f1"]) == ["1", "2", "3", "4", "5", "6", "7"]
    assert np.asarray(metrics["confusion_matrix"]).shape == (7, 7)


def test_metric_report_handles_missing_test_labels_without_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        metrics = classification_metrics(
            y_true=np.asarray([1, 1, 2, 2]),
            y_pred=np.asarray([1, 7, 2, 7]),
            y_train=np.asarray([1, 2, 3, 7]),
        )

    warning_messages = [str(warning.message) for warning in caught_warnings]
    assert not any("y_pred contains classes not in y_true" in message for message in warning_messages)
    assert np.asarray(metrics["confusion_matrix"]).shape == (7, 7)
    assert metrics["labels_missing_from_test"] == [3, 4, 5, 6, 7]
    assert metrics["labels_missing_from_train"] == [4, 5, 6]
    assert metrics["labels_predicted_not_in_test"] == [7]
    assert metrics["test_label_distribution"]["7"] == 0
    assert metrics["test_gesture_distribution"]["extended_palm"] == 0
    assert len(metrics["class_imbalance_warnings"]) == 7


def test_baseline_training_runs_on_tiny_synthetic_dataset() -> None:
    workspace = _test_workspace()
    try:
        dataset = _synthetic_window_dataset()
        windows_path = workspace / "data" / "processed" / "emg_windows.npz"
        results_path = workspace / "reports" / "metrics" / "baseline_results.json"
        figures_dir = workspace / "reports" / "figures"
        models_dir = workspace / "models"
        windows_path.parent.mkdir(parents=True)
        np.savez_compressed(windows_path, **dataset)

        report = run_training(
            windows_path=windows_path,
            results_path=results_path,
            figures_dir=figures_dir,
            models_dir=models_dir,
            test_size=0.25,
            random_state=42,
        )

        assert results_path.exists()
        assert (figures_dir / "confusion_matrix_random_split.png").exists()
        assert (figures_dir / "confusion_matrix_subject_split.png").exists()
        assert (models_dir / "baseline_random_forest.joblib").exists()

        loaded_report = json.loads(results_path.read_text(encoding="utf-8"))
        assert loaded_report["label_order"] == [1, 2, 3, 4, 5, 6, 7]
        assert loaded_report["gesture_mapping"]["7"] == "extended_palm"
        assert loaded_report["splits"]["random_split"]["random_forest"]["test_size"] > 0
        assert (
            np.asarray(
                loaded_report["splits"]["random_split"]["random_forest"][
                    "confusion_matrix"
                ]
            ).shape
            == (7, 7)
        )
        assert "train_label_distribution" in loaded_report["splits"]["random_split"][
            "random_forest"
        ]
        assert "labels_missing_from_test" in loaded_report["splits"]["subject_split"][
            "random_forest"
        ]
        assert loaded_report == report
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
