"""Build the processed UCI EMG dataset and metadata summary."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.parse_uci_emg import (  # noqa: E402
    DEFAULT_RAW_DIR,
    ParseResult,
    parse_raw_dataset_with_report,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path("configs/data.yaml")
DEFAULT_OUTPUT_PATH = Path("data/processed/emg_samples.parquet")
DEFAULT_SUMMARY_PATH = Path("reports/metrics/dataset_summary.json")


def load_data_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Load data configuration when a YAML config file is available."""
    path = Path(config_path)
    if not path.exists():
        return {}

    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not config:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in data config: {path}")
    return config


def create_dataset_summary(parse_result: ParseResult) -> dict[str, object]:
    """Create a JSON-serializable dataset summary."""
    dataframe = parse_result.dataframe

    if dataframe.empty:
        label_distribution: dict[str, int] = {}
        rows_per_subject: dict[str, int] = {}
        number_of_subjects = 0
        number_of_recordings = 0
    else:
        label_distribution = {
            str(label_id): int(count)
            for label_id, count in dataframe["label_id"]
            .value_counts()
            .sort_index()
            .items()
        }
        rows_per_subject = {
            str(subject_id): int(count)
            for subject_id, count in dataframe["subject_id"]
            .value_counts()
            .sort_index()
            .items()
        }
        number_of_subjects = int(dataframe["subject_id"].nunique())
        number_of_recordings = int(dataframe["source_file"].nunique())

    return {
        "total_rows": int(len(dataframe)),
        "number_of_subjects": number_of_subjects,
        "number_of_recordings": number_of_recordings,
        "label_distribution": label_distribution,
        "rows_per_subject": rows_per_subject,
        "files_parsed": parse_result.files_parsed,
        "files_failed": parse_result.files_failed,
    }


def save_summary(summary: dict[str, object], summary_path: str | Path) -> None:
    """Write a dataset summary JSON file."""
    output_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def save_processed_dataset(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    """Write the processed dataset to parquet."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataframe.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Writing parquet requires pyarrow or fastparquet. "
            "Install project dependencies from requirements.txt."
        ) from exc


def print_summary(summary: dict[str, object]) -> None:
    """Print a compact terminal summary."""
    print("Dataset summary")
    print(f"  Total rows: {summary['total_rows']}")
    print(f"  Subjects: {summary['number_of_subjects']}")
    print(f"  Recordings: {summary['number_of_recordings']}")
    print(f"  Files parsed: {len(summary['files_parsed'])}")
    print(f"  Files failed: {len(summary['files_failed'])}")
    print("  Label distribution:")
    label_distribution = summary["label_distribution"]
    if isinstance(label_distribution, dict) and label_distribution:
        for label_id, count in label_distribution.items():
            print(f"    {label_id}: {count}")
    else:
        print("    none")


def build_dataset(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    keep_unmarked: bool = False,
) -> dict[str, object]:
    """Parse raw files, save parquet output, and write a summary JSON."""
    LOGGER.info("Parsing raw dataset from %s", raw_dir)
    parse_result = parse_raw_dataset_with_report(
        raw_dir=raw_dir,
        keep_unmarked=keep_unmarked,
    )
    save_processed_dataset(parse_result.dataframe, output_path)

    summary = create_dataset_summary(parse_result)
    save_summary(summary, summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--keep-unmarked",
        action="store_true",
        help="Keep rows with label_id 0 instead of dropping them.",
    )
    return parser.parse_args()


def resolve_build_options(args: argparse.Namespace) -> dict[str, object]:
    """Resolve CLI arguments against config defaults."""
    config = load_data_config(args.config)
    drop_unmarked = bool(config.get("drop_unmarked", True))

    return {
        "raw_dir": args.raw_dir or Path(config.get("raw_data_dir", DEFAULT_RAW_DIR)),
        "output_path": args.output
        or Path(config.get("processed_samples_path", DEFAULT_OUTPUT_PATH)),
        "summary_path": args.summary
        or Path(config.get("summary_path", DEFAULT_SUMMARY_PATH)),
        "keep_unmarked": args.keep_unmarked or not drop_unmarked,
    }


def main() -> None:
    """Run the dataset builder CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    summary = build_dataset(**resolve_build_options(args))
    print_summary(summary)


if __name__ == "__main__":
    main()
