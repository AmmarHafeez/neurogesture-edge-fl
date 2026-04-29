"""Export a trained CNN-1D checkpoint to ONNX with preprocessing metadata."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import EXPECTED_LABEL_ORDER, GESTURE_MAPPING  # noqa: E402
from src.evaluation.splits import make_subject_split  # noqa: E402
from src.models.cnn1d import CNN1D  # noqa: E402
from src.preprocessing.normalization import normalize_windows_for_split  # noqa: E402
from src.training.train_deep import load_window_dataset  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path("models/cnn1d_subject_split_best.pt")
DEFAULT_ONNX_PATH = Path("models/onnx/cnn1d_fp32.onnx")
DEFAULT_WINDOWS_PATH = Path("data/processed/emg_windows.npz")
DEFAULT_INPUT_SHAPE = (1, 8, 200)


def _import_onnx():
    """Import ONNX or raise a clear dependency error."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export validation requires the 'onnx' package. "
            "Install project dependencies from requirements.txt."
        ) from exc
    return onnx


def metadata_path_for_model(model_path: str | Path) -> Path:
    """Return the metadata JSON path next to an ONNX model."""
    path = Path(model_path)
    return path.with_name(f"{path.stem}_metadata.json")


def load_cnn_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[CNN1D, dict[str, object]]:
    """Load a CNN-1D model and checkpoint dictionary."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CNN checkpoint does not exist: {path}. Run train_deep.py first."
        )

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    input_channels = int(checkpoint.get("input_channels", 8))
    model = CNN1D(input_channels=input_channels, num_classes=len(EXPECTED_LABEL_ORDER))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def normalization_metadata_from_windows(
    windows_path: str | Path,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, object]:
    """Compute global-channel normalization metadata from global training subjects."""
    dataset = load_window_dataset(windows_path)
    X = dataset["X"].astype(np.float32)
    subject_ids = dataset["subject_ids"].astype(str)
    train_idx, _, _ = make_subject_split(
        subject_ids=subject_ids,
        test_size=test_size,
        random_state=random_state,
    )
    _, metadata = normalize_windows_for_split(
        X=X,
        train_idx=train_idx,
        mode="global_channel_zscore",
    )
    return metadata


def resolve_normalization_metadata(
    checkpoint: dict[str, object],
    windows_path: str | Path | None = None,
) -> dict[str, object]:
    """Resolve normalization metadata from checkpoint or window data."""
    checkpoint_metadata = checkpoint.get("normalization")
    if isinstance(checkpoint_metadata, dict):
        return checkpoint_metadata

    if windows_path is None:
        raise ValueError(
            "Checkpoint does not contain normalization metadata. "
            "Pass --windows to compute global_channel_zscore metadata."
        )
    return normalization_metadata_from_windows(windows_path)


def save_metadata(metadata: dict[str, object], metadata_path: str | Path) -> None:
    """Save ONNX metadata JSON."""
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def export_cnn_to_onnx(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    output_path: str | Path = DEFAULT_ONNX_PATH,
    windows_path: str | Path | None = DEFAULT_WINDOWS_PATH,
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    dynamic_batch: bool = True,
    opset_version: int = 17,
    device_name: str = "cpu",
) -> dict[str, object]:
    """Export a CNN-1D checkpoint to ONNX and save metadata."""
    onnx = _import_onnx()
    device = torch.device(device_name)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_cnn_checkpoint(checkpoint_path, device=device)
    normalization_metadata = resolve_normalization_metadata(
        checkpoint,
        windows_path=windows_path,
    )

    dummy_input = torch.zeros(input_shape, dtype=torch.float32, device=device)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "emg_window": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ["emg_window"],
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
        "dynamo": False,
    }
    try:
        torch.onnx.export(model, dummy_input, output, **export_kwargs)
    except TypeError as exc:
        if "dynamo" not in str(exc):
            raise
        export_kwargs.pop("dynamo")
        torch.onnx.export(model, dummy_input, output, **export_kwargs)
    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)

    channel_names = [f"emg_{index}" for index in range(1, input_shape[1] + 1)]
    metadata = {
        "model_name": str(checkpoint.get("model_name", "cnn1d")),
        "input_shape": list(input_shape),
        "output_shape": [input_shape[0], len(EXPECTED_LABEL_ORDER)],
        "label_order": EXPECTED_LABEL_ORDER,
        "gesture_mapping": {str(k): v for k, v in GESTURE_MAPPING.items()},
        "normalization_mode": normalization_metadata["normalization_mode"],
        "normalization_mean": normalization_metadata["normalization_mean"],
        "normalization_std": normalization_metadata["normalization_std"],
        "window_size": input_shape[2],
        "channel_names": channel_names,
        "checkpoint_path": str(checkpoint_path),
        "onnx_path": str(output),
    }
    save_metadata(metadata, metadata_path_for_model(output))
    LOGGER.info("Exported ONNX model to %s", output)
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS_PATH)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--static-batch", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run ONNX export from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    metadata = export_cnn_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        windows_path=args.windows,
        input_shape=(args.batch_size, args.channels, args.window_size),
        dynamic_batch=not args.static_batch,
        opset_version=args.opset_version,
    )
    print("ONNX export complete")
    print(f"  ONNX: {metadata['onnx_path']}")
    print(f"  Metadata: {metadata_path_for_model(args.output)}")


if __name__ == "__main__":
    main()
