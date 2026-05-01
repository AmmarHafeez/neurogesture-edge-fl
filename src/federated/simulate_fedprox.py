"""Run a lightweight subject-level FedProx simulation for EMG windows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.reports import save_json_report  # noqa: E402
from src.federated.simulate_fedavg import (  # noqa: E402
    DEFAULT_MODELS_DIR,
    DEFAULT_WINDOWS_PATH,
    run_federated_simulation,
)
from src.training.train_deep import load_window_dataset  # noqa: E402


DEFAULT_RESULTS_PATH = Path("reports/metrics/fedprox_results.json")


def run_simulation(
    windows_path: str | Path = DEFAULT_WINDOWS_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    rounds: int = 5,
    clients_per_round: int = 8,
    local_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    mu: float = 0.01,
    random_state: int = 42,
    model_name: str = "cnn1d",
    test_size: float = 0.2,
    use_class_weights: bool = True,
    device_name: str | None = None,
    save_final_checkpoint: bool = True,
) -> dict[str, object]:
    """Load EMG windows, run FedProx, and save the result JSON."""
    dataset = load_window_dataset(windows_path)
    report = run_federated_simulation(
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
        aggregation="fedprox",
        fedprox_mu=mu,
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
    parser.add_argument("--mu", type=float, default=0.01)
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
        help="Do not save the final global FedProx checkpoint.",
    )
    parser.set_defaults(use_class_weights=True, save_final_checkpoint=True)
    return parser.parse_args()


def main() -> None:
    """Run the FedProx CLI."""
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
        mu=args.mu,
        random_state=args.random_state,
        model_name=args.model,
        test_size=args.test_size,
        use_class_weights=args.use_class_weights,
        device_name=args.device,
        save_final_checkpoint=args.save_final_checkpoint,
    )
    print("Federated FedProx simulation complete")
    print(f"  Clients: {report['number_of_clients']}")
    print(f"  Rounds: {report['rounds']}")
    print(f"  FedProx mu: {report['fedprox_mu']}")
    print(f"  Final macro F1: {report['final_macro_f1']:.4f}")
    print(f"  Final balanced accuracy: {report['final_balanced_accuracy']:.4f}")
    print(f"  Results: {args.results}")


if __name__ == "__main__":
    main()
