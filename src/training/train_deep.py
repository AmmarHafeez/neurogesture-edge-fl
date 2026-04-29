"""Train lightweight PyTorch deep baselines on EMG windows."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (  # noqa: E402
    EXPECTED_LABEL_ORDER,
    GESTURE_MAPPING,
    classification_metrics,
    label_distribution,
)
from src.evaluation.reports import save_json_report  # noqa: E402
from src.evaluation.splits import make_random_split, make_subject_split  # noqa: E402
from src.models.cnn1d import CNN1D  # noqa: E402
from src.models.tcn import TCN  # noqa: E402
from src.preprocessing.normalization import normalize_windows_for_split  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_WINDOWS_PATH = Path("data/processed/emg_windows.npz")
DEFAULT_RESULTS_PATH = Path("reports/metrics/deep_results.json")
DEFAULT_MODELS_DIR = Path("models")


def set_seed(seed: int) -> None:
    """Set random seeds for local PyTorch experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labels_to_zero_based(labels: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert external labels 1..7 to internal labels 0..6."""
    return labels - 1


def labels_to_external(labels: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert internal labels 0..6 back to external labels 1..7."""
    return labels + 1


class EMGWindowDataset(Dataset):
    """Torch dataset for EMG windows stored as numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError("X must have shape (n_windows, window_size, n_channels)")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of windows")

        self.X = np.asarray(X, dtype=np.float32)
        self.y_external = np.asarray(y, dtype=np.int64)
        if not np.isin(self.y_external, EXPECTED_LABEL_ORDER).all():
            raise ValueError("y must contain labels in the range 1..7")
        self.y_internal = labels_to_zero_based(self.y_external).astype(np.int64)

    def __len__(self) -> int:
        return len(self.y_internal)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.from_numpy(self.X[index]).transpose(0, 1).contiguous()
        label = torch.tensor(self.y_internal[index], dtype=torch.long)
        return window, label


def load_window_dataset(windows_path: str | Path) -> dict[str, np.ndarray]:
    """Load compressed EMG windows for deep training."""
    path = Path(windows_path)
    if not path.exists():
        raise FileNotFoundError(f"Window dataset does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Window dataset path is not a file: {path}")

    with np.load(path, allow_pickle=False) as loaded:
        required_keys = {"X", "y", "subject_ids"}
        missing_keys = sorted(required_keys - set(loaded.files))
        if missing_keys:
            raise ValueError(f"Window dataset is missing keys: {missing_keys}")
        return {key: loaded[key] for key in loaded.files}


def build_model(
    model_name: str,
    input_channels: int = 8,
    num_classes: int = 7,
) -> nn.Module:
    """Build a supported deep baseline model."""
    if model_name == "cnn1d":
        return CNN1D(input_channels=input_channels, num_classes=num_classes)
    if model_name == "tcn":
        return TCN(input_channels=input_channels, num_classes=num_classes)
    raise ValueError(f"Unsupported deep model: {model_name}")


def compute_class_weights(y_train_external: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    labels = np.asarray(y_train_external, dtype=np.int64)
    counts = np.asarray(
        [np.sum(labels == label) for label in EXPECTED_LABEL_ORDER],
        dtype=np.float32,
    )
    weights = np.zeros_like(counts, dtype=np.float32)
    present = counts > 0
    if present.any():
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
    weights[~present] = 0.0
    return torch.from_numpy(weights)


def make_dataloader(
    dataset: Dataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a dataloader over a dataset subset."""
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 5.0,
) -> float:
    """Train for one epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for windows, labels in dataloader:
        windows = windows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(windows)
        loss = criterion(logits, labels)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        batch_size = len(labels)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataloader is empty")
    return total_loss / total_samples


@torch.no_grad()
def predict_external_labels(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Predict external 1..7 labels for a dataloader."""
    model.eval()
    predictions: list[np.ndarray] = []
    for windows, _ in dataloader:
        logits = model(windows.to(device))
        batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.append(labels_to_external(batch_predictions))
    if not predictions:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(predictions).astype(np.int64)


def majority_class_baseline_metrics(
    y_train: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Evaluate a majority-class baseline from the training labels."""
    labels, counts = np.unique(y_train, return_counts=True)
    majority_label = int(labels[np.argmax(counts)])
    predictions = np.full_like(y_true, fill_value=majority_label)
    metrics = classification_metrics(
        y_true=y_true,
        y_pred=predictions,
        y_train=y_train,
    )
    return {
        "majority_class_label": majority_label,
        "majority_class_baseline_accuracy": float(metrics["accuracy"]),
        "majority_class_baseline_macro_f1": float(metrics["macro_f1"]),
    }


def train_model_for_split(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_path: str | Path,
    normalization_mode: str = "global_channel_zscore",
    use_class_weights: bool = True,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 5.0,
    random_state: int = 42,
    test_subjects: list[str] | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    """Train one deep model on one split and return fitted model plus metrics."""
    set_seed(random_state)
    X_normalized, normalization_metadata = normalize_windows_for_split(
        X=X,
        train_idx=train_idx,
        mode=normalization_mode,
    )
    dataset = EMGWindowDataset(X_normalized, y)
    train_loader = make_dataloader(dataset, train_idx, batch_size=batch_size, shuffle=True)
    train_eval_loader = make_dataloader(
        dataset,
        train_idx,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = make_dataloader(dataset, test_idx, batch_size=batch_size, shuffle=False)

    input_channels = int(X.shape[2])
    model = build_model(
        model_name=model_name,
        input_channels=input_channels,
        num_classes=len(EXPECTED_LABEL_ORDER),
    ).to(device)

    class_weights = compute_class_weights(y[train_idx]).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights if use_class_weights else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_macro_f1 = -1.0
    best_epoch = 0
    history: list[dict[str, object]] = []

    y_test = y[test_idx]
    y_train = y[train_idx]
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        train_predictions = predict_external_labels(
            model,
            train_eval_loader,
            device=device,
        )
        train_metrics = classification_metrics(
            y_true=y_train,
            y_pred=train_predictions,
            y_train=y_train,
        )
        evaluation_predictions = predict_external_labels(
            model,
            test_loader,
            device=device,
        )
        evaluation_metrics = classification_metrics(
            y_true=y_test,
            y_pred=evaluation_predictions,
            y_train=y_train,
            test_subjects=test_subjects,
        )
        evaluation_macro_f1 = float(evaluation_metrics["macro_f1"])
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "evaluation_accuracy": evaluation_metrics["accuracy"],
            "evaluation_macro_f1": evaluation_metrics["macro_f1"],
            "evaluation_balanced_accuracy": evaluation_metrics["balanced_accuracy"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_summary)
        scheduler.step(evaluation_macro_f1)

        if evaluation_macro_f1 > best_macro_f1:
            best_macro_f1 = evaluation_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    final_predictions = predict_external_labels(model, test_loader, device=device)
    final_metrics = classification_metrics(
        y_true=y_test,
        y_pred=final_predictions,
        y_train=y_train,
        test_subjects=test_subjects,
    )
    final_metrics["normalization_mode"] = normalization_metadata["normalization_mode"]
    final_metrics["normalization_mean"] = normalization_metadata["normalization_mean"]
    final_metrics["normalization_std"] = normalization_metadata["normalization_std"]
    final_metrics["class_weights_enabled"] = use_class_weights
    final_metrics["weight_decay"] = weight_decay
    final_metrics["max_grad_norm"] = max_grad_norm
    final_metrics["predicted_label_distribution"] = label_distribution(
        final_predictions
    )
    final_metrics["true_label_distribution"] = final_metrics["test_label_distribution"]
    final_metrics.update(
        majority_class_baseline_metrics(
            y_train=y_train,
            y_true=y_test,
        )
    )
    final_metrics["best_epoch"] = best_epoch
    final_metrics["best_validation_macro_f1"] = best_macro_f1
    final_metrics["training_history"] = history

    checkpoint = {
        "model_name": model_name,
        "model_state_dict": best_state,
        "label_order": EXPECTED_LABEL_ORDER,
        "gesture_mapping": GESTURE_MAPPING,
        "input_channels": input_channels,
        "normalization": normalization_metadata,
    }
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_file)

    return model, final_metrics


def train_deep_baseline(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    model_name: str = "cnn1d",
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    device_name: str | None = None,
    normalization_mode: str = "global_channel_zscore",
    use_class_weights: bool = True,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 5.0,
) -> dict[str, object]:
    """Train and evaluate a deep baseline on random and subject splits."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    y = np.asarray(y, dtype=np.int64)
    subject_ids = np.asarray(subject_ids).astype(str)

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

    models_path = Path(models_dir)
    checkpoint_name = f"{model_name}_best.pt"
    random_checkpoint_path = models_path / checkpoint_name

    _, random_metrics = train_model_for_split(
        model_name=model_name,
        X=X,
        y=y,
        train_idx=random_train_idx,
        test_idx=random_test_idx,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=random_checkpoint_path,
        normalization_mode=normalization_mode,
        use_class_weights=use_class_weights,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        random_state=random_state,
    )
    _, subject_metrics = train_model_for_split(
        model_name=model_name,
        X=X,
        y=y,
        train_idx=subject_train_idx,
        test_idx=subject_test_idx,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=models_path / f"{model_name}_subject_split_best.pt",
        normalization_mode=normalization_mode,
        use_class_weights=use_class_weights,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        random_state=random_state,
        test_subjects=test_subjects,
    )

    return {
        "model_name": model_name,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "class_weights_enabled": use_class_weights,
        "normalization_mode": normalization_mode,
        "normalization_mean": {
            "random_split": random_metrics["normalization_mean"],
            "subject_split": subject_metrics["normalization_mean"],
        },
        "normalization_std": {
            "random_split": random_metrics["normalization_std"],
            "subject_split": subject_metrics["normalization_std"],
        },
        "label_order": EXPECTED_LABEL_ORDER,
        "gesture_mapping": {str(k): v for k, v in GESTURE_MAPPING.items()},
        "best_epoch": {
            "random_split": random_metrics["best_epoch"],
            "subject_split": subject_metrics["best_epoch"],
        },
        "best_validation_macro_f1": {
            "random_split": random_metrics["best_validation_macro_f1"],
            "subject_split": subject_metrics["best_validation_macro_f1"],
        },
        "training_history": {
            "random_split": random_metrics["training_history"],
            "subject_split": subject_metrics["training_history"],
        },
        "splits": {
            "random_split": random_metrics,
            "subject_split": subject_metrics,
        },
    }


def run_training(
    windows_path: str | Path = DEFAULT_WINDOWS_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    model_name: str = "cnn1d",
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
    device_name: str | None = None,
    normalization_mode: str = "global_channel_zscore",
    use_class_weights: bool = True,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 5.0,
) -> dict[str, object]:
    """Load window data, train a deep baseline, and save results."""
    dataset = load_window_dataset(windows_path)
    report = train_deep_baseline(
        X=dataset["X"],
        y=dataset["y"].astype(int),
        subject_ids=dataset["subject_ids"].astype(str),
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        test_size=test_size,
        random_state=random_state,
        models_dir=models_dir,
        device_name=device_name,
        normalization_mode=normalization_mode,
        use_class_weights=use_class_weights,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )
    save_json_report(report, results_path)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model", choices=["cnn1d", "tcn"], default="cnn1d")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--normalization",
        choices=["global_channel_zscore", "per_window_channel_zscore"],
        default="global_channel_zscore",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_false",
        dest="use_class_weights",
        help="Disable class weights in CrossEntropyLoss.",
    )
    parser.set_defaults(use_class_weights=True)
    return parser.parse_args()


def main() -> None:
    """Run the deep training CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    report = run_training(
        windows_path=args.windows,
        results_path=args.results,
        models_dir=args.models_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        random_state=args.random_state,
        device_name=args.device,
        normalization_mode=args.normalization,
        use_class_weights=args.use_class_weights,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )
    print("Deep baseline evaluation complete")
    print(f"  Model: {report['model_name']}")
    print(f"  Device: {report['device']}")
    print(f"  Results: {args.results}")
    print(f"  Models: {args.models_dir}")
    for split_name, metrics in report["splits"].items():
        print(
            f"  {split_name}: best_epoch={metrics['best_epoch']} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
