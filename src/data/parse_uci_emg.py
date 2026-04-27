"""Parser for the UCI EMG Data for Gestures raw text files."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/EMG_data_for_gestures-master")

EMG_COLUMNS = [
    "emg_1",
    "emg_2",
    "emg_3",
    "emg_4",
    "emg_5",
    "emg_6",
    "emg_7",
    "emg_8",
]

COLUMN_NAMES = [
    "time",
    *EMG_COLUMNS,
    "label_id",
]

OUTPUT_COLUMNS = [
    *COLUMN_NAMES,
    "subject_id",
    "source_file",
    "recording_id",
    "gesture",
]

GESTURE_LABELS = {
    0: "unmarked",
    1: "rest",
    2: "fist",
    3: "wrist_flexion",
    4: "wrist_extension",
    5: "radial_deviation",
    6: "ulnar_deviation",
    7: "extended_palm",
}


@dataclass(frozen=True)
class ParseResult:
    """Parsed dataset plus per-file parse status."""

    dataframe: pd.DataFrame
    files_parsed: list[str]
    files_failed: dict[str, str]


def discover_raw_files(raw_dir: str | Path = DEFAULT_RAW_DIR) -> list[Path]:
    """Discover raw text files below the dataset root.

    Files directly under the dataset root are skipped so dataset README files are
    not treated as recordings.
    """
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Raw dataset path is not a directory: {root}")

    return sorted(
        path
        for path in root.rglob("*.txt")
        if path.is_file() and path.parent != root
    )


def extract_subject_id(path: str | Path) -> str:
    """Extract the subject identifier from a raw file parent folder."""
    subject_id = Path(path).parent.name.strip()
    if not subject_id:
        raise ValueError(f"Cannot extract subject_id from path: {path}")
    return subject_id


def extract_recording_id(path: str | Path) -> str:
    """Extract the recording identifier from a raw file name."""
    recording_id = Path(path).stem.strip()
    if not recording_id:
        raise ValueError(f"Cannot extract recording_id from path: {path}")
    return recording_id


def _read_raw_table(path: Path) -> pd.DataFrame:
    """Read a whitespace- or tab-separated raw text file."""
    try:
        table = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=COLUMN_NAMES,
            dtype=str,
            engine="python",
            on_bad_lines="skip",
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Raw file is empty: {path}") from exc

    table = table.dropna(how="all")
    return table


def _coerce_numeric_columns(dataframe: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Coerce expected columns and drop non-numeric rows."""
    parsed = dataframe.copy()
    original_rows = len(parsed)

    parsed[COLUMN_NAMES] = parsed[COLUMN_NAMES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    valid_numeric_rows = parsed[COLUMN_NAMES].notna().all(axis=1)
    dropped_non_numeric = int((~valid_numeric_rows).sum())
    parsed = parsed.loc[valid_numeric_rows].copy()

    if dropped_non_numeric:
        LOGGER.info(
            "Dropped %s non-numeric rows from %s",
            dropped_non_numeric,
            path,
        )

    if not (parsed["label_id"] % 1 == 0).all():
        before_integer_filter = len(parsed)
        parsed = parsed.loc[parsed["label_id"] % 1 == 0].copy()
        dropped_non_integer = before_integer_filter - len(parsed)
        LOGGER.info(
            "Dropped %s rows with non-integer labels from %s",
            dropped_non_integer,
            path,
        )

    parsed["label_id"] = parsed["label_id"].astype(int)
    known_label_rows = parsed["label_id"].isin(GESTURE_LABELS)
    unknown_labels = sorted(set(parsed.loc[~known_label_rows, "label_id"]))
    if unknown_labels:
        LOGGER.info("Dropped rows with unknown label ids in %s: %s", path, unknown_labels)
        parsed = parsed.loc[known_label_rows].copy()

    LOGGER.info(
        "Parsed %s: input_rows=%s valid_numeric_rows=%s dropped_non_numeric=%s",
        path,
        original_rows,
        len(parsed),
        dropped_non_numeric,
    )

    return parsed


def parse_emg_file(path: str | Path, keep_unmarked: bool = False) -> pd.DataFrame:
    """Parse one UCI EMG raw text file into a canonical dataframe."""
    raw_path = Path(path)
    dataframe = _read_raw_table(raw_path)
    dataframe = _coerce_numeric_columns(dataframe, raw_path)

    if not keep_unmarked:
        rows_before_drop = len(dataframe)
        dataframe = dataframe[dataframe["label_id"] != 0].copy()
        dropped_unmarked = rows_before_drop - len(dataframe)
        if dropped_unmarked:
            LOGGER.info("Dropped %s unmarked rows from %s", dropped_unmarked, raw_path)

    dataframe["subject_id"] = extract_subject_id(raw_path)
    dataframe["source_file"] = raw_path.as_posix()
    dataframe["recording_id"] = extract_recording_id(raw_path)
    dataframe["gesture"] = dataframe["label_id"].map(GESTURE_LABELS)

    return dataframe[OUTPUT_COLUMNS].reset_index(drop=True)


def read_emg_text_file(path: str | Path) -> pd.DataFrame:
    """Read one raw EMG text file into a dataframe with canonical columns."""
    return parse_emg_file(path)


def parse_raw_dataset_with_report(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    keep_unmarked: bool = False,
) -> ParseResult:
    """Parse all discovered raw files and keep per-file parse status."""
    frames: list[pd.DataFrame] = []
    files_parsed: list[str] = []
    files_failed: dict[str, str] = {}

    for raw_file in discover_raw_files(raw_dir):
        source_file = raw_file.as_posix()
        try:
            dataframe = parse_emg_file(raw_file, keep_unmarked=keep_unmarked)
            frames.append(dataframe)
            files_parsed.append(source_file)
            LOGGER.info("File parsed: %s rows=%s", raw_file, len(dataframe))
        except Exception as exc:  # noqa: BLE001 - failures are recorded in summary.
            LOGGER.warning("Failed to parse %s: %s", raw_file, exc)
            files_failed[source_file] = str(exc)

    if frames:
        dataframe = pd.concat(frames, ignore_index=True)
    else:
        dataframe = pd.DataFrame(columns=OUTPUT_COLUMNS)

    return ParseResult(
        dataframe=dataframe,
        files_parsed=files_parsed,
        files_failed=files_failed,
    )


def parse_raw_dataset(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    keep_unmarked: bool = False,
) -> pd.DataFrame:
    """Parse all discovered raw files and return the combined dataframe."""
    return parse_raw_dataset_with_report(
        raw_dir=raw_dir,
        keep_unmarked=keep_unmarked,
    ).dataframe
