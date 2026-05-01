"""Run a lightweight subject-level FedAvg simulation for EMG windows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (  # noqa: E402
    EXPECTED_LABEL_ORDER,
    GESTURE_MAPPING,
    classification_metrics,
)
from src.evaluation.reports import save_json_report  # noqa: E402
from src.evaluation.splits import make_subject_split  # noqa: E402
from src.federated.aggregation import fedavg_state_dict  # noqa: E402
from src.federated.client import ClientUpdate, FederatedClient, build_subject_clients  # noqa: E402
from src.preprocessing.normalization import normalize_windows_for_split  # noqa: E402
from src.training.train_deep import (  # noqa: E402
    EMGWindowDataset,
    build_model,
    compute_class_weights,
    load_window_dataset,
    make_dataloader,
    predict_external_labels,
    set_seed,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_WINDOWS_PATH = Path("data/processed/emg_windows.npz")
DEFAULT_RESULTS_PATH = Path("reports/metrics/federated_results.json")
DEFAULT_MODELS_DIR = Path("models")


def select_clients(
    client_ids: list[str],
    clients_per_round: int,
    round_number: int,
    random_state: int = 42,
) -> list[str]:
    """Select a deterministic subset of client IDs for one round."""
    if clients_per_round <= 0:
        raise ValueError("clients_per_round must be positive")
    if not client_ids:
        raise ValueError("At least one client is required")

    ordered_client_ids = sorted(str(client_id) for client_id in client_ids)
    if clients_per_round >= len(ordered_client_ids):
        return ordered_client_ids

    rng = np.random.default_rng(random_state + round_number)
    selected = rng.choice(
        ordered_client_ids,
        size=clients_per_round,
        replace=False,
    )
    return sorted(str(client_id) for client_id in selected.tolist())


def evaluate_global_model(
    model: nn.Module,
    X_normalized: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    device: torch.device,
    test_subjects: list[str],
) -> dict[str, object]:
    """Evaluate the current global model on held-out subject windows."""
    dataset = EMGWindowDataset(X_normalized, y)
    test_loader = make_dataloader(
        dataset,
        indices=np.asarray(test_idx, dtype=int),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = predict_external_labels(model, test_loader, device=device)
    return classification_metrics(
        y_true=y[test_idx],
        y_pred=predictions,
        y_train=y[train_idx],
        test_subjects=test_subjects,
    )


def train_selected_clients(
    global_model: nn.Module,
    clients: dict[str, FederatedClient],
    selected_client_ids: list[str],
    criterion: nn.Module,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    random_state: int,
    round_number: int,
    weight_decay: float,
    max_grad_norm: float,
    fedprox_mu: float = 0.0,
) -> list[ClientUpdate]:
    """Train selected clients from the current global model."""
    updates: list[ClientUpdate] = []
    for offset, subject_id in enumerate(selected_client_ids):
        client = clients[subject_id]
        update = client.train(
            global_model=global_model,
            criterion=criterion,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            random_state=random_state + round_number * 10_000 + offset,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            fedprox_mu=fedprox_mu,
        )
        updates.append(update)
    return updates


def run_federated_simulation(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    rounds: int = 5,
    clients_per_round: int = 8,
    local_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    model_name: str = "cnn1d",
    test_size: float = 0.2,
    use_class_weights: bool = True,
    device_name: str | None = None,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 5.0,
    models_dir: str | Path | None = DEFAULT_MODELS_DIR,
    save_final_checkpoint: bool = True,
    aggregation: str = "fedavg",
    fedprox_mu: float = 0.0,
) -> dict[str, object]:
    """Run a manual federated simulation where each subject is one client."""
    if aggregation not in {"fedavg", "fedprox"}:
        raise ValueError("aggregation must be 'fedavg' or 'fedprox'")
    if fedprox_mu < 0:
        raise ValueError("fedprox_mu must be non-negative")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if local_epochs <= 0:
        raise ValueError("local_epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    set_seed(random_state)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    values = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y, dtype=np.int64)
    subjects = np.asarray(subject_ids).astype(str)

    train_idx, test_idx, held_out_subjects = make_subject_split(
        subject_ids=subjects,
        test_size=test_size,
        random_state=random_state,
    )
    training_subjects = sorted(np.unique(subjects[train_idx]).astype(str).tolist())
    X_normalized, normalization_metadata = normalize_windows_for_split(
        X=values,
        train_idx=train_idx,
        mode="global_channel_zscore",
    )
    clients = build_subject_clients(
        X=X_normalized,
        y=labels,
        subject_ids=subjects,
        train_idx=train_idx,
    )

    model = build_model(
        model_name=model_name,
        input_channels=int(values.shape[2]),
        num_classes=len(EXPECTED_LABEL_ORDER),
    ).to(device)

    class_weights = compute_class_weights(labels[train_idx]).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights if use_class_weights else None,
    )

    round_history: list[dict[str, object]] = []
    best_round_by_macro_f1 = 0
    best_macro_f1 = -1.0
    best_balanced_accuracy = 0.0

    client_ids = sorted(clients.keys())
    for round_number in range(1, rounds + 1):
        selected_clients = select_clients(
            client_ids=client_ids,
            clients_per_round=clients_per_round,
            round_number=round_number,
            random_state=random_state,
        )
        LOGGER.info(
            "Federated round %s/%s selected %s clients",
            round_number,
            rounds,
            len(selected_clients),
        )
        updates = train_selected_clients(
            global_model=model,
            clients=clients,
            selected_client_ids=selected_clients,
            criterion=criterion,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            random_state=random_state,
            round_number=round_number,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            fedprox_mu=fedprox_mu if aggregation == "fedprox" else 0.0,
        )
        aggregated_state = fedavg_state_dict(
            [(update.state_dict, update.num_samples) for update in updates]
        )
        model.load_state_dict(aggregated_state)
        model.to(device)

        evaluation_metrics = evaluate_global_model(
            model=model,
            X_normalized=X_normalized,
            y=labels,
            train_idx=train_idx,
            test_idx=test_idx,
            batch_size=batch_size,
            device=device,
            test_subjects=held_out_subjects,
        )
        total_client_samples = int(sum(update.num_samples for update in updates))
        mean_client_loss = float(
            np.average(
                [update.mean_loss for update in updates],
                weights=[update.num_samples for update in updates],
            )
        )
        round_summary = {
            "round": round_number,
            "selected_clients": selected_clients,
            "number_of_selected_clients": len(selected_clients),
            "total_client_samples": total_client_samples,
            "mean_client_loss": mean_client_loss,
            "evaluation_metrics": evaluation_metrics,
        }
        round_history.append(round_summary)

        macro_f1 = float(evaluation_metrics["macro_f1"])
        if macro_f1 > best_macro_f1:
            best_round_by_macro_f1 = round_number
            best_macro_f1 = macro_f1
            best_balanced_accuracy = float(evaluation_metrics["balanced_accuracy"])

    final_metrics = round_history[-1]["evaluation_metrics"]
    checkpoint_path = None
    if save_final_checkpoint and models_dir is not None:
        output_dir = Path(models_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"{aggregation}_{model_name}_final.pt"
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "label_order": EXPECTED_LABEL_ORDER,
                "gesture_mapping": GESTURE_MAPPING,
                "input_channels": int(values.shape[2]),
                "normalization": normalization_metadata,
                "aggregation": aggregation,
                "fedprox_mu": fedprox_mu if aggregation == "fedprox" else 0.0,
                "rounds": rounds,
            },
            checkpoint_path,
        )

    report: dict[str, object] = {
        "experiment": f"federated_{aggregation}",
        "aggregation": aggregation,
        "model_name": model_name,
        "device": str(device),
        "rounds": rounds,
        "clients_per_round": clients_per_round,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "random_state": random_state,
        "class_weights_enabled": use_class_weights,
        "training_subjects": training_subjects,
        "held_out_subjects": [str(subject) for subject in held_out_subjects],
        "number_of_clients": len(clients),
        "normalization_mode": normalization_metadata["normalization_mode"],
        "normalization_mean": normalization_metadata["normalization_mean"],
        "normalization_std": normalization_metadata["normalization_std"],
        "label_order": EXPECTED_LABEL_ORDER,
        "gesture_mapping": {str(key): value for key, value in GESTURE_MAPPING.items()},
        "round_history": round_history,
        "final_macro_f1": float(final_metrics["macro_f1"]),
        "final_balanced_accuracy": float(final_metrics["balanced_accuracy"]),
        "best_round_by_macro_f1": best_round_by_macro_f1,
        "best_macro_f1": best_macro_f1,
        "best_balanced_accuracy": best_balanced_accuracy,
    }
    if aggregation == "fedprox":
        report["fedprox_mu"] = fedprox_mu
    if checkpoint_path is not None:
        report["final_checkpoint_path"] = str(checkpoint_path)
    return report


def run_fedavg_simulation(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    rounds: int = 5,
    clients_per_round: int = 8,
    local_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    model_name: str = "cnn1d",
    test_size: float = 0.2,
    use_class_weights: bool = True,
    device_name: str | None = None,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 5.0,
    models_dir: str | Path | None = DEFAULT_MODELS_DIR,
    save_final_checkpoint: bool = True,
) -> dict[str, object]:
    """Run manual FedAvg where each subject is one simulated client."""
    return run_federated_simulation(
        X=X,
        y=y,
        subject_ids=subject_ids,
        rounds=rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        model_name=model_name,
        test_size=test_size,
        use_class_weights=use_class_weights,
        device_name=device_name,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        models_dir=models_dir,
        save_final_checkpoint=save_final_checkpoint,
        aggregation="fedavg",
        fedprox_mu=0.0,
    )


def run_simulation(
    windows_path: str | Path = DEFAULT_WINDOWS_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    rounds: int = 5,
    clients_per_round: int = 8,
    local_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
    model_name: str = "cnn1d",
    test_size: float = 0.2,
    use_class_weights: bool = True,
    device_name: str | None = None,
    save_final_checkpoint: bool = True,
) -> dict[str, object]:
    """Load EMG windows, run FedAvg, and save the result JSON."""
    dataset = load_window_dataset(windows_path)
    report = run_fedavg_simulation(
        X=dataset["X"],
        y=dataset["y"].astype(int),
        subject_ids=dataset["subject_ids"].astype(str),
        rounds=rounds,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
        model_name=model_name,
        test_size=test_size,
        use_class_weights=use_class_weights,
        device_name=device_name,
        models_dir=models_dir,
        save_final_checkpoint=save_final_checkpoint,
    )
    save_json_report(report, results_path)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model", choices=["cnn1d"], default="cnn1d")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients-per-round", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--no-class-weights",
        action="store_false",
        dest="use_class_weights",
        help="Disable class weights in local client CrossEntropyLoss.",
    )
    parser.add_argument(
        "--no-final-checkpoint",
        action="store_false",
        dest="save_final_checkpoint",
        help="Do not save the final global FedAvg checkpoint.",
    )
    parser.set_defaults(use_class_weights=True, save_final_checkpoint=True)
    return parser.parse_args()


def main() -> None:
    """Run the FedAvg CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    report = run_simulation(
        windows_path=args.windows,
        results_path=args.results,
        models_dir=args.models_dir,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        model_name=args.model,
        test_size=args.test_size,
        use_class_weights=args.use_class_weights,
        device_name=args.device,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    print("Federated FedAvg simulation complete")
    print(f"  Clients: {report['number_of_clients']}")
    print(f"  Rounds: {report['rounds']}")
    print(f"  Final macro F1: {report['final_macro_f1']:.4f}")
    print(f"  Final balanced accuracy: {report['final_balanced_accuracy']:.4f}")
    print(f"  Results: {args.results}")


if __name__ == "__main__":
    main()
