"""Quantize an ONNX model with ONNX Runtime dynamic INT8 quantization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.edge.export_onnx import metadata_path_for_model, save_metadata


LOGGER = logging.getLogger(__name__)

DEFAULT_FP32_PATH = Path("models/onnx/cnn1d_fp32.onnx")
DEFAULT_INT8_PATH = Path("models/onnx/cnn1d_int8.onnx")


def _import_quantization():
    """Import ONNX Runtime quantization utilities or raise a clear error."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError(
            "ONNX INT8 quantization requires 'onnxruntime'. "
            "Install project dependencies from requirements.txt."
        ) from exc
    return quantize_dynamic, QuantType


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def quantize_onnx_model(
    input_path: str | Path = DEFAULT_FP32_PATH,
    output_path: str | Path = DEFAULT_INT8_PATH,
) -> dict[str, object]:
    """Apply dynamic INT8 quantization to an ONNX model."""
    quantize_dynamic, QuantType = _import_quantization()
    source = Path(input_path)
    target = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"Input ONNX model does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        quantize_dynamic(
            model_input=str(source),
            model_output=str(target),
            weight_type=QuantType.QInt8,
        )
    except Exception as exc:
        if not target.exists():
            raise RuntimeError(f"INT8 quantization failed and no output was created: {exc}") from exc
        LOGGER.warning("INT8 quantization reported an error but output exists: %s", exc)

    if not target.exists():
        raise RuntimeError(f"INT8 quantization did not create output model: {target}")

    metadata = _load_metadata(metadata_path_for_model(source))
    metadata.update(
        {
            "quantization": "dynamic_int8",
            "input_onnx_path": str(source),
            "onnx_path": str(target),
            "output_onnx_path": str(target),
        }
    )
    save_metadata(metadata, metadata_path_for_model(target))
    LOGGER.info("Quantized ONNX model to %s", target)
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_FP32_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_INT8_PATH)
    return parser.parse_args()


def main() -> None:
    """Run ONNX quantization from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    metadata = quantize_onnx_model(input_path=args.input, output_path=args.output)
    print("ONNX quantization complete")
    print(f"  ONNX: {metadata['onnx_path']}")
    print(f"  Metadata: {metadata_path_for_model(args.output)}")


if __name__ == "__main__":
    main()
