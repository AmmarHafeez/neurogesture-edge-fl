"""Evaluate user calibration for a held-out-subject CNN baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.evaluation.reports import save_json_report  # noqa: E402
from src.evaluation.splits import make_subject_split  # noqa: E402
from src.personalization.calibrate_user import (  # noqa: E402
    build_cnn_from_checkpoint,
    fine_tune_model,
    make_eval_loader,
    sample_calibration_indices,
)
from src.preprocessing.normalization import normalize_windows_for_split  # noqa: E402
from src.training.train_deep import (  # noqa: E402
    load_window_dataset,
    predict_external_labels,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_WINDOWS_PATH = Path("data/processed/emg_windows.npz")
DEFAULT_BASE_MODEL_PATH = Path("models/cnn1d_subject_split_best.pt")
DEFAULT_RESULTS_PATH = Path("reports/metrics/personalization_results.json")


def _mean(values: list[float]) -> float:
    """Return a JSON-friendly mean with a zero fallback."""
    return float(np.mean(values)) if values else 0.0


def aggregate_personalization_results(
    subject_results: list[dict[str, object]],
    skipped_subjects: list[dict[str, object]],
) -> dict[str, object]:
    """Aggregate per-subject personalization metrics."""
    before_macro = [
        float(result["before_personalization"]["macro_f1"])
        for result in subject_results
    ]
    after_macro = [
        float(result["after_personalization"]["macro_f1"])
        for result in subject_results
    ]
    delta_macro = [float(result["delta_macro_f1"]) for result in subject_results]

    before_balanced = [
        float(result["before_personalization"]["balanced_accuracy"])
        for result in subject_results
    ]
    after_balanced = [
        float(result["after_personalization"]["balanced_accuracy"])
        for result in subject_results
    ]
    delta_balanced = [
        float(result["delta_balanced_accuracy"])
        for result in subject_results
    ]

    return {
        "mean_before_macro_f1": _mean(before_macro),
        "mean_after_macro_f1": _mean(after_macro),
        "mean_delta_macro_f1": _mean(delta_macro),
        "mean_before_balanced_accuracy": _mean(before_balanced),
        "mean_after_balanced_accuracy": _mean(after_balanced),
        "mean_delta_balanced_accuracy": _mean(delta_balanced),
        "number_of_subjects_evaluated": len(subject_results),
        "number_of_subjects_skipped": len(skipped_subjects),
    }


def evaluate_subject_calibration(
    subject_id: str,
    subject_indices: np.ndarray,
    base_model: torch.nn.Module,
    X_normalized: np.ndarray,
    y: np.ndarray,
    calibration_per_class: int,
    mode: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    random_state: int = 42,
) -> dict[str, object]:
    """Run before/after personalization evaluation for one target subject."""
    calibration_idx, evaluation_idx, calibration_counts, evaluation_counts = (
        sample_calibration_indices(
            subject_indices=subject_indices,
            labels=y,
            calibration_per_class=calibration_per_class,
            random_state=random_state,
        )
    )
    if len(evaluation_idx) == 0:
        return {
            "subject_id": subject_id,
            "skipped": True,
            "warning": "No evaluation windows remain after calibration sampling.",
            "calibration_size": int(len(calibration_idx)),
            "evaluation_size": 0,
            "calibration_label_distribution": calibration_counts,
            "evaluation_label_distribution": evaluation_counts,
        }

    eval_loader = make_eval_loader(
        X_normalized=X_normalized,
        y=y,
        indices=evaluation_idx,
        batch_size=batch_size,
    )
    y_eval = y[evaluation_idx]
    y_calibration = y[calibration_idx]

    before_predictions = predict_external_labels(base_model, eval_loader, device=device)
    before_metrics = classification_metrics(
        y_true=y_eval,
        y_pred=before_predictions,
        y_train=y_calibration,
    )

    personalized_model = fine_tune_model(
        base_model=base_model,
        X_normalized=X_normalized,
        y=y,
        calibration_idx=calibration_idx,
        mode=mode,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        random_state=random_state,
    )
    after_predictions = predict_external_labels(
        personalized_model,
        eval_loader,
        device=device,
    )
    after_metrics = classification_metrics(
        y_true=y_eval,
        y_pred=after_predictions,
        y_train=y_calibration,
    )

    return {
        "subject_id": subject_id,
        "skipped": False,
        "calibration_size": int(len(calibration_idx)),
        "evaluation_size": int(len(evaluation_idx)),
        "calibration_label_distribution": calibration_counts,
        "evaluation_label_distribution": evaluation_counts,
        "before_personalization": before_metrics,
        "after_personalization": after_metrics,
        "delta_macro_f1": float(after_metrics["macro_f1"] - before_metrics["macro_f1"]),
        "delta_balanced_accuracy": float(
            after_metrics["balanced_accuracy"] - before_metrics["balanced_accuracy"]
        ),
        "calibration_mode": mode,
        "personalization_epochs": epochs,
    }


def run_personalization_experiment(
    windows_path: str | Path = DEFAULT_WINDOWS_PATH,
    base_model_path: str | Path = DEFAULT_BASE_MODEL_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    mode: str = "last_layer",
    calibration_per_class: int = 10,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    test_size: float = 0.2,
    random_state: int = 42,
    device_name: str | None = None,
) -> dict[str, object]:
    """Run personalization for held-out subjects and save a JSON report."""
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset = load_window_dataset(windows_path)
    X = dataset["X"].astype(np.float32)
    y = dataset["y"].astype(int)
    subject_ids = dataset["subject_ids"].astype(str)

    train_idx, test_idx, held_out_subjects = make_subject_split(
        subject_ids=subject_ids,
        test_size=test_size,
        random_state=random_state,
    )
    X_normalized, normalization_metadata = normalize_windows_for_split(
        X=X,
        train_idx=train_idx,
        mode="global_channel_zscore",
    )

    base_model = build_cnn_from_checkpoint(
        checkpoint_path=base_model_path,
        input_channels=int(X.shape[2]),
        device=device,
    )
    base_model.eval()

    subject_results: list[dict[str, object]] = []
    skipped_subjects: list[dict[str, object]] = []
    test_index_set = set(test_idx.tolist())
    for subject_id in held_out_subjects:
        subject_indices = np.asarray(
            [
                index
                for index in np.flatnonzero(subject_ids == str(subject_id)).tolist()
                if index in test_index_set
            ],
            dtype=int,
        )
        result = evaluate_subject_calibration(
            subject_id=str(subject_id),
            subject_indices=subject_indices,
            base_model=base_model,
            X_normalized=X_normalized,
            y=y,
            calibration_per_class=calibration_per_class,
            mode=mode,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            random_state=random_state,
        )
        if result.get("skipped"):
            LOGGER.warning("Skipping subject %s: %s", subject_id, result["warning"])
            skipped_subjects.append(result)
        else:
            subject_results.append(result)

    report = {
        "experiment": "personalized_calibration",
        "base_model": str(base_model_path),
        "device": str(device),
        "normalization_mode": normalization_metadata["normalization_mode"],
        "normalization_mean": normalization_metadata["normalization_mean"],
        "normalization_std": normalization_metadata["normalization_std"],
        "calibration_mode": mode,
        "calibration_per_class": calibration_per_class,
        "personalization_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "held_out_subjects": [str(subject) for subject in held_out_subjects],
        "subjects": subject_results,
        "skipped_subjects": skipped_subjects,
        "aggregate": aggregate_personalization_results(
            subject_results=subject_results,
            skipped_subjects=skipped_subjects,
        ),
    }
    save_json_report(report, results_path)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS_PATH)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--mode", choices=["last_layer", "full_model"], default="last_layer")
    parser.add_argument("--calibration-per-class", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def main() -> None:
    """Run the personalization CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    report = run_personalization_experiment(
        windows_path=args.windows,
        base_model_path=args.base_model,
        results_path=args.results,
        mode=args.mode,
        calibration_per_class=args.calibration_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        random_state=args.random_state,
        device_name=args.device,
    )
    aggregate = report["aggregate"]
    print("Personalization evaluation complete")
    print(f"  Results: {args.results}")
    print(f"  Subjects evaluated: {aggregate['number_of_subjects_evaluated']}")
    print(f"  Subjects skipped: {aggregate['number_of_subjects_skipped']}")
    print(f"  Mean delta macro F1: {aggregate['mean_delta_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
