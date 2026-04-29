from pathlib import Path
import shutil
import uuid

import numpy as np
import torch
from torch import nn

from src.federated.aggregation import fedavg_state_dict
from src.federated.client import FederatedClient, build_subject_clients
from src.federated.simulate_fedavg import run_simulation, select_clients
from src.models.cnn1d import CNN1D


def _test_workspace() -> Path:
    path = Path("tests/generated") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def _synthetic_federated_data(
    subjects: int = 4,
    repeats_per_label: int = 2,
) -> dict[str, np.ndarray]:
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
    for subject in range(1, subjects + 1):
        for label in range(1, 8):
            for repeat in range(repeats_per_label):
                value = subject + label * 0.1 + repeat * 0.01
                X.append(np.full((24, 8), fill_value=value, dtype=np.float32))
                y.append(label)
                subject_ids.append(str(subject))
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


def test_fedavg_weighted_aggregation_uses_sample_counts() -> None:
    state_a = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor(4, dtype=torch.long),
    }
    state_b = {
        "weight": torch.tensor([3.0, 7.0]),
        "counter": torch.tensor(9, dtype=torch.long),
    }

    aggregated = fedavg_state_dict([(state_a, 1), (state_b, 3)])

    torch.testing.assert_close(aggregated["weight"], torch.tensor([2.5, 6.0]))
    assert aggregated["counter"].item() == 4


def test_client_datasets_contain_one_subject_each() -> None:
    data = _synthetic_federated_data(subjects=3, repeats_per_label=1)
    subject_ids = data["subject_ids"]
    train_idx = np.flatnonzero(subject_ids != "3")

    clients = build_subject_clients(
        X=data["X"],
        y=data["y"],
        subject_ids=subject_ids,
        train_idx=train_idx,
    )

    assert sorted(clients.keys()) == ["1", "2"]
    for subject_id, client in clients.items():
        expected_count = int(np.sum(subject_ids[train_idx] == subject_id))
        assert client.subject_id == subject_id
        assert client.num_samples == expected_count


def test_client_selection_is_deterministic() -> None:
    client_ids = ["1", "2", "3", "4", "5"]

    first = select_clients(
        client_ids,
        clients_per_round=3,
        round_number=2,
        random_state=42,
    )
    second = select_clients(
        client_ids,
        clients_per_round=3,
        round_number=2,
        random_state=42,
    )

    assert first == second
    assert len(first) == 3


def test_one_local_client_training_step_runs() -> None:
    X = np.random.default_rng(42).normal(size=(14, 24, 8)).astype(np.float32)
    y = np.asarray([1, 2, 3, 4, 5, 6, 7] * 2, dtype=np.int64)
    client = FederatedClient(subject_id="1", X=X, y=y)
    model = CNN1D(input_channels=8, num_classes=7)
    criterion = nn.CrossEntropyLoss()

    update = client.train(
        global_model=model,
        criterion=criterion,
        local_epochs=1,
        batch_size=7,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        random_state=42,
    )

    assert update.subject_id == "1"
    assert update.num_samples == 14
    assert update.mean_loss > 0
    assert set(update.state_dict) == set(model.state_dict())


def test_federated_simulation_smoke_run_writes_results() -> None:
    workspace = _test_workspace()
    try:
        data = _synthetic_federated_data(subjects=4, repeats_per_label=2)
        windows_path = workspace / "data" / "processed" / "emg_windows.npz"
        results_path = workspace / "reports" / "metrics" / "federated_results.json"
        models_dir = workspace / "models"
        windows_path.parent.mkdir(parents=True)
        np.savez_compressed(windows_path, **data)

        report = run_simulation(
            windows_path=windows_path,
            results_path=results_path,
            models_dir=models_dir,
            rounds=1,
            clients_per_round=2,
            local_epochs=1,
            batch_size=7,
            learning_rate=1e-3,
            random_state=42,
            device_name="cpu",
        )

        assert results_path.exists()
        assert report["experiment"] == "federated_fedavg"
        assert report["aggregation"] == "fedavg"
        assert len(report["round_history"]) == 1
        assert "evaluation_metrics" in report["round_history"][0]
        assert "final_macro_f1" in report
        assert "final_balanced_accuracy" in report
        assert report["number_of_clients"] == 3
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
