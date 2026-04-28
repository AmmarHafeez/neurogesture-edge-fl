"""Train and evaluate classical baseline models on EMG windows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import joblib
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (  # noqa: E402
    EXPECTED_LABEL_ORDER,
    GESTURE_MAPPING,
    classification_metrics,
)
from src.evaluation.reports import (  # noqa: E402
    save_confusion_matrix_figure,
    save_json_report,
)
from src.evaluation.splits import make_random_split, make_subject_split  # noqa: E402
from src.models.classical import (  # noqa: E402
    build_classical_models,
    extract_window_features,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_WINDOWS_PATH = Path("data/processed/emg_windows.npz")
DEFAULT_RESULTS_PATH = Path("reports/metrics/baseline_results.json")
DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_MODELS_DIR = Path("models")

REQUIRED_WINDOW_KEYS = {
    "X",
    "y",
    "subject_ids",
    "recording_ids",
    "gesture_names",
    "channel_names",
    "window_size",
    "stride",
}


def load_window_dataset(windows_path: str | Path) -> dict[str, np.ndarray]:
    """Load a compressed EMG window dataset."""
    path = Path(windows_path)
    if not path.exists():
        raise FileNotFoundError(f"Window dataset does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Window dataset path is not a file: {path}")

    with np.load(path, allow_pickle=False) as loaded:
        missing_keys = sorted(REQUIRED_WINDOW_KEYS - set(loaded.files))
        if missing_keys:
            raise ValueError(f"Window dataset is missing keys: {missing_keys}")

        dataset = {key: loaded[key] for key in loaded.files}
    X = dataset["X"]
    y = dataset["y"]
    subject_ids = dataset["subject_ids"]
    if X.ndim != 3:
        raise ValueError("X must have shape (n_windows, window_size, n_channels)")
    if len(X) != len(y) or len(X) != len(subject_ids):
        raise ValueError("X, y, and subject_ids must contain the same number of rows")
    return dataset


def fit_and_score_model(
    model: object,
    X_features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: np.ndarray,
    test_subjects: list[str] | None = None,
) -> tuple[object, dict[str, object]]:
    """Fit one model and return the fitted model plus metrics."""
    X_train = X_features[train_idx]
    X_test = X_features[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = classification_metrics(
        y_true=y_test,
        y_pred=predictions,
        labels=labels,
        y_train=y_train,
        train_size=len(train_idx),
        test_subjects=test_subjects,
    )
    return model, metrics


def _best_model_name_for_split(results: dict[str, dict[str, object]]) -> str:
    """Select the model with the highest macro F1 for one split."""
    return max(
        results,
        key=lambda name: float(results[name].get("macro_f1", 0.0)),
    )


def train_baselines(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_mode: str = "flatten",
) -> tuple[dict[str, object], object]:
    """Train and evaluate baseline models on random and subject splits."""
    X_features = extract_window_features(X, mode=feature_mode)
    labels = np.asarray(EXPECTED_LABEL_ORDER, dtype=int)

    random_train_idx, random_test_idx = make_random_split(
        y=y,
        test_size=test_size,
        random_state=random_state,
    )
    subject_train_idx, subject_test_idx, test_subjects = make_subject_split(
        subject_ids=subject_ids,
        test_size=test_size,
        random_state=random_state,
    )

    report: dict[str, object] = {
        "feature_mode": feature_mode,
        "labels": labels.tolist(),
        "label_order": labels.tolist(),
        "gesture_mapping": {
            str(label): gesture for label, gesture in GESTURE_MAPPING.items()
        },
        "splits": {
            "random_split": {},
            "subject_split": {},
        },
    }
    fitted_random_forest = None

    split_specs = {
        "random_split": (random_train_idx, random_test_idx, None),
        "subject_split": (subject_train_idx, subject_test_idx, test_subjects),
    }
    for split_name, (train_idx, test_idx, split_subjects) in split_specs.items():
        LOGGER.info(
            "Evaluating %s with train_size=%s test_size=%s",
            split_name,
            len(train_idx),
            len(test_idx),
        )
        split_results: dict[str, dict[str, object]] = {}
        for model_name, model in build_classical_models(
            random_state=random_state
        ).items():
            LOGGER.info("Training %s on %s", model_name, split_name)
            fitted_model, metrics = fit_and_score_model(
                model=model,
                X_features=X_features,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                labels=labels,
                test_subjects=split_subjects,
            )
            split_results[model_name] = metrics
            if split_name == "random_split" and model_name == "random_forest":
                fitted_random_forest = fitted_model

        split_results["best_model_by_macro_f1"] = _best_model_name_for_split(
            split_results
        )
        report["splits"][split_name] = split_results

    if fitted_random_forest is None:
        raise RuntimeError("Random forest baseline was not trained")

    return report, fitted_random_forest


def save_baseline_outputs(
    report: dict[str, object],
    random_forest_model: object,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> None:
    """Persist baseline metrics, confusion matrix figures, and model artifact."""
    save_json_report(report, results_path)

    labels = report["label_order"]
    figures_path = Path(figures_dir)
    splits = report["splits"]
    for split_name, filename in {
        "random_split": "confusion_matrix_random_split.png",
        "subject_split": "confusion_matrix_subject_split.png",
    }.items():
        split_results = splits[split_name]
        best_model_name = split_results["best_model_by_macro_f1"]
        confusion = split_results[best_model_name]["confusion_matrix"]
        save_confusion_matrix_figure(
            confusion=confusion,
            labels=labels,
            output_path=figures_path / filename,
            title=f"{split_name}: {best_model_name}",
        )

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(random_forest_model, models_path / "baseline_random_forest.joblib")


def run_training(
    windows_path: str | Path = DEFAULT_WINDOWS_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    feature_mode: str = "flatten",
) -> dict[str, object]:
    """Load windows, train baselines, and save outputs."""
    dataset = load_window_dataset(windows_path)
    report, random_forest_model = train_baselines(
        X=dataset["X"],
        y=dataset["y"].astype(int),
        subject_ids=dataset["subject_ids"].astype(str),
        test_size=test_size,
        random_state=random_state,
        feature_mode=feature_mode,
    )
    save_baseline_outputs(
        report=report,
        random_forest_model=random_forest_model,
        results_path=results_path,
        figures_dir=figures_dir,
        models_dir=models_dir,
    )
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--feature-mode",
        choices=["flatten", "mean_absolute_value"],
        default="flatten",
    )
    return parser.parse_args()


def main() -> None:
    """Run the baseline training CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    report = run_training(
        windows_path=args.windows,
        results_path=args.results,
        figures_dir=args.figures_dir,
        models_dir=args.models_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        feature_mode=args.feature_mode,
    )
    print("Baseline evaluation complete")
    print(f"  Results: {args.results}")
    print(f"  Models: {args.models_dir}")
    for split_name, split_results in report["splits"].items():
        best_model = split_results["best_model_by_macro_f1"]
        macro_f1 = split_results[best_model]["macro_f1"]
        balanced_accuracy = split_results[best_model]["balanced_accuracy"]
        print(
            f"  {split_name}: best={best_model} "
            f"macro_f1={macro_f1:.4f} balanced_accuracy={balanced_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
