"""Benchmark ONNX inference latency for edge deployment."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np


LOGGER = logging.getLogger(__name__)

DEFAULT_FP32_PATH = Path("models/onnx/cnn1d_fp32.onnx")
DEFAULT_INT8_PATH = Path("models/onnx/cnn1d_int8.onnx")
DEFAULT_OUTPUT_PATH = Path("reports/metrics/edge_benchmark.json")
DEFAULT_INPUT_SHAPE = (1, 8, 200)


def _import_onnxruntime():
    """Import ONNX Runtime or raise a clear dependency error."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX latency benchmarking requires 'onnxruntime'. "
            "Install project dependencies from requirements.txt."
        ) from exc
    return ort


def model_size_kb(model_path: str | Path) -> float:
    """Return model file size in kilobytes."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    return path.stat().st_size / 1024.0


def summarize_latency_samples(
    latency_ms: list[float] | np.ndarray,
    model_size: float,
    providers: list[str],
) -> dict[str, object]:
    """Summarize latency samples and model size."""
    values = np.asarray(latency_ms, dtype=float)
    if values.size == 0:
        raise ValueError("latency_ms must contain at least one value")
    mean_latency = float(values.mean())
    return {
        "model_size_kb": float(model_size),
        "mean_latency_ms": mean_latency,
        "median_latency_ms": float(np.median(values)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "min_latency_ms": float(values.min()),
        "max_latency_ms": float(values.max()),
        "throughput_windows_per_second": float(1000.0 / mean_latency)
        if mean_latency > 0
        else 0.0,
        "runtime_providers": providers,
    }


def benchmark_onnx_model(
    model_path: str | Path,
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    warmup: int = 20,
    runs: int = 200,
) -> dict[str, object]:
    """Benchmark one ONNX model with random float32 input."""
    if runs <= 0 or warmup < 0:
        raise ValueError("runs must be positive and warmup must be non-negative")
    ort = _import_onnxruntime()
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {path}")

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    sample = np.random.default_rng(42).normal(size=input_shape).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: sample})

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        latencies.append((time.perf_counter() - start) * 1000.0)

    return summarize_latency_samples(
        latency_ms=latencies,
        model_size=model_size_kb(path),
        providers=session.get_providers(),
    )


def benchmark_models(
    fp32_model: str | Path = DEFAULT_FP32_PATH,
    int8_model: str | Path | None = DEFAULT_INT8_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    warmup: int = 20,
    runs: int = 200,
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
) -> dict[str, object]:
    """Benchmark FP32 and optional INT8 ONNX models and save JSON results."""
    results: dict[str, object] = {
        "input_shape": list(input_shape),
        "warmup": warmup,
        "runs": runs,
        "models": {
            "fp32": benchmark_onnx_model(
                fp32_model,
                input_shape=input_shape,
                warmup=warmup,
                runs=runs,
            )
        },
    }

    if int8_model is not None:
        int8_path = Path(int8_model)
        if int8_path.exists():
            results["models"]["int8"] = benchmark_onnx_model(
                int8_path,
                input_shape=input_shape,
                warmup=warmup,
                runs=runs,
            )
        else:
            LOGGER.warning("INT8 model not found; skipping benchmark: %s", int8_path)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-model", type=Path, default=DEFAULT_FP32_PATH)
    parser.add_argument("--int8-model", type=Path, default=DEFAULT_INT8_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """Run ONNX latency benchmarking from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    benchmark_models(
        fp32_model=args.fp32_model,
        int8_model=args.int8_model,
        output_path=args.output,
        warmup=args.warmup,
        runs=args.runs,
    )
    print("Edge benchmark complete")
    print(f"  Results: {args.output}")


if __name__ == "__main__":
    main()
