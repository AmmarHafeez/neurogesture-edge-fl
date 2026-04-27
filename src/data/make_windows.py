"""Build fixed-length EMG windows from parsed sample parquet data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.preprocessing.windowing import (  # noqa: E402
    WindowedDataset,
    build_windows_from_dataframe,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = Path("data/processed/emg_samples.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/emg_windows.npz")
DEFAULT_SUMMARY_PATH = Path("reports/metrics/window_summary.json")
DEFAULT_WINDOW_SIZE = 200
DEFAULT_STRIDE = 100


def load_samples(input_path: str | Path) -> pd.DataFrame:
    """Load parsed EMG sample rows from parquet."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Parsed EMG sample file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Parsed EMG sample path is not a file: {path}")
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet requires pyarrow or fastparquet. "
            "Install project dependencies from requirements.txt."
        ) from exc


def save_windowed_dataset(windowed_dataset: WindowedDataset, output_path: str | Path) -> None:
    """Save window arrays and metadata to a compressed NPZ file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=windowed_dataset.X,
        y=windowed_dataset.y,
        subject_ids=windowed_dataset.subject_ids,
        recording_ids=windowed_dataset.recording_ids,
        gesture_names=windowed_dataset.gesture_names,
        channel_names=windowed_dataset.channel_names,
        window_size=np.asarray(windowed_dataset.window_size, dtype=np.int64),
        stride=np.asarray(windowed_dataset.stride, dtype=np.int64),
    )


def create_window_summary(
    input_rows: int,
    windowed_dataset: WindowedDataset,
) -> dict[str, object]:
    """Create a JSON-serializable summary for generated windows."""
    number_of_windows = int(len(windowed_dataset.y))
    subject_ids = windowed_dataset.subject_ids
    recording_ids = windowed_dataset.recording_ids
    gesture_names = windowed_dataset.gesture_names
    recording_pairs = set(zip(subject_ids.tolist(), recording_ids.tolist(), strict=True))

    label_distribution = {
        str(label): int(count)
        for label, count in zip(
            *np.unique(windowed_dataset.y, return_counts=True),
            strict=True,
        )
    }
    gesture_distribution = {
        str(gesture): int(count)
        for gesture, count in zip(
            *np.unique(gesture_names, return_counts=True),
            strict=True,
        )
    }
    windows_per_subject = {
        str(subject_id): int(count)
        for subject_id, count in zip(
            *np.unique(subject_ids, return_counts=True),
            strict=True,
        )
    }

    if windows_per_subject:
        per_subject_counts = np.asarray(list(windows_per_subject.values()), dtype=float)
        min_windows = int(per_subject_counts.min())
        max_windows = int(per_subject_counts.max())
        mean_windows = float(per_subject_counts.mean())
    else:
        min_windows = 0
        max_windows = 0
        mean_windows = 0.0

    return {
        "input_rows": int(input_rows),
        "window_size": int(windowed_dataset.window_size),
        "stride": int(windowed_dataset.stride),
        "number_of_windows": number_of_windows,
        "number_of_subjects": int(len(np.unique(subject_ids))),
        "number_of_recordings": int(len(recording_pairs)),
        "label_distribution_windows": label_distribution,
        "gesture_distribution_windows": gesture_distribution,
        "windows_per_subject": windows_per_subject,
        "min_windows_per_subject": min_windows,
        "max_windows_per_subject": max_windows,
        "mean_windows_per_subject": mean_windows,
    }


def save_summary(summary: dict[str, object], summary_path: str | Path) -> None:
    """Write a window summary JSON file."""
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def print_summary(summary: dict[str, object]) -> None:
    """Print a compact terminal summary."""
    print("Window summary")
    print(f"  Input rows: {summary['input_rows']}")
    print(f"  Window size: {summary['window_size']}")
    print(f"  Stride: {summary['stride']}")
    print(f"  Windows: {summary['number_of_windows']}")
    print(f"  Subjects: {summary['number_of_subjects']}")
    print(f"  Recordings: {summary['number_of_recordings']}")


def build_windows_file(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> dict[str, object]:
    """Load parsed samples, save window arrays, and write a summary."""
    LOGGER.info("Loading parsed EMG samples from %s", input_path)
    dataframe = load_samples(input_path)

    LOGGER.info(
        "Building windows with window_size=%s stride=%s",
        window_size,
        stride,
    )
    windowed_dataset = build_windows_from_dataframe(
        dataframe,
        window_size=window_size,
        stride=stride,
    )

    save_windowed_dataset(windowed_dataset, output_path)
    LOGGER.info("Saved %s windows to %s", len(windowed_dataset.y), output_path)

    summary = create_window_summary(
        input_rows=len(dataframe),
        windowed_dataset=windowed_dataset,
    )
    save_summary(summary, summary_path)
    LOGGER.info("Saved window summary to %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    return parser.parse_args()


def main() -> None:
    """Run the window builder CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    summary = build_windows_file(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        window_size=args.window_size,
        stride=args.stride,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
