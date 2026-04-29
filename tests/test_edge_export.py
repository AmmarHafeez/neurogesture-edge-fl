from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
import torch

from src.edge.benchmark_latency import model_size_kb, summarize_latency_samples
from src.edge.export_onnx import export_cnn_to_onnx, metadata_path_for_model
from src.models.cnn1d import CNN1D


def _test_workspace() -> Path:
    path = Path("tests/generated") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_cnn_export_to_temporary_onnx_file() -> None:
    onnx = pytest.importorskip("onnx")
    workspace = _test_workspace()
    try:
        checkpoint_path = workspace / "models" / "cnn1d_subject_split_best.pt"
        onnx_path = workspace / "models" / "onnx" / "cnn1d_fp32.onnx"
        checkpoint_path.parent.mkdir(parents=True)
        model = CNN1D(input_channels=8, num_classes=7)
        torch.save(
            {
                "model_name": "cnn1d",
                "model_state_dict": model.state_dict(),
                "input_channels": 8,
                "normalization": {
                    "normalization_mode": "global_channel_zscore",
                    "normalization_mean": [0.0] * 8,
                    "normalization_std": [1.0] * 8,
                },
            },
            checkpoint_path,
        )

        metadata = export_cnn_to_onnx(
            checkpoint_path=checkpoint_path,
            output_path=onnx_path,
            windows_path=None,
            input_shape=(1, 8, 200),
        )

        assert onnx_path.exists()
        onnx.checker.check_model(onnx.load(onnx_path))
        assert metadata["input_shape"] == [1, 8, 200]
        assert metadata["label_order"] == [1, 2, 3, 4, 5, 6, 7]
        assert metadata["gesture_mapping"]["7"] == "extended_palm"
        assert metadata["normalization_mode"] == "global_channel_zscore"
        assert metadata_path_for_model(onnx_path).exists()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_model_size_helper_returns_kilobytes() -> None:
    workspace = _test_workspace()
    try:
        model_path = workspace / "model.onnx"
        model_path.write_bytes(b"0" * 2048)

        assert model_size_kb(model_path) == 2.0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_latency_summary_from_synthetic_timings() -> None:
    summary = summarize_latency_samples(
        latency_ms=np.asarray([1.0, 2.0, 3.0, 4.0]),
        model_size=12.5,
        providers=["CPUExecutionProvider"],
    )

    assert summary["model_size_kb"] == 12.5
    assert summary["mean_latency_ms"] == 2.5
    assert summary["median_latency_ms"] == 2.5
    assert summary["min_latency_ms"] == 1.0
    assert summary["max_latency_ms"] == 4.0
    assert summary["throughput_windows_per_second"] == 400.0
    assert summary["runtime_providers"] == ["CPUExecutionProvider"]
