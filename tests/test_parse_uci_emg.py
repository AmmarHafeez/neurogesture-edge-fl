import json
import logging
from pathlib import Path
import shutil
import uuid

from src.data.make_dataset import create_dataset_summary, save_summary
from src.data.parse_uci_emg import (
    OUTPUT_COLUMNS,
    parse_emg_file,
    parse_raw_dataset_with_report,
)


def _test_workspace() -> Path:
    path = Path("tests/generated") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_parse_synthetic_file_without_header() -> None:
    workspace = _test_workspace()
    subject_dir = workspace / "EMG_data_for_gestures-master" / "1"
    subject_dir.mkdir(parents=True)
    raw_file = subject_dir / "trial_a.txt"
    try:
        raw_file.write_text(
            "\n".join(
                [
                    "0.00 1 2 3 4 5 6 7 8 0",
                    "0.01\t2\t3\t4\t5\t6\t7\t8\t9\t1",
                    "0.02 3 4 5 6 7 8 9 10 2",
                ]
            ),
            encoding="utf-8",
        )

        dataframe = parse_emg_file(raw_file)

        assert list(dataframe.columns) == OUTPUT_COLUMNS
        assert len(dataframe) == 2
        assert dataframe["subject_id"].unique().tolist() == ["1"]
        assert dataframe["recording_id"].unique().tolist() == ["trial_a"]
        assert dataframe["label_id"].tolist() == [1, 2]
        assert dataframe["gesture"].tolist() == ["rest", "fist"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_parse_synthetic_file_with_header_row() -> None:
    workspace = _test_workspace()
    subject_dir = workspace / "EMG_data_for_gestures-master" / "4"
    subject_dir.mkdir(parents=True)
    raw_file = subject_dir / "trial_with_header.txt"
    try:
        raw_file.write_text(
            "\n".join(
                [
                    "time emg_1 emg_2 emg_3 emg_4 emg_5 emg_6 emg_7 emg_8 label_id",
                    "0.00 1 2 3 4 5 6 7 8 1",
                    "0.01 2 3 4 5 6 7 8 9 3",
                ]
            ),
            encoding="utf-8",
        )

        dataframe = parse_emg_file(raw_file)

        assert len(dataframe) == 2
        assert dataframe["subject_id"].unique().tolist() == ["4"]
        assert dataframe["recording_id"].unique().tolist() == ["trial_with_header"]
        assert dataframe["label_id"].tolist() == [1, 3]
        assert dataframe["gesture"].tolist() == ["rest", "wrist_flexion"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_parse_synthetic_file_with_one_invalid_non_numeric_row(caplog) -> None:
    workspace = _test_workspace()
    subject_dir = workspace / "EMG_data_for_gestures-master" / "5"
    subject_dir.mkdir(parents=True)
    raw_file = subject_dir / "trial_with_invalid_row.txt"
    try:
        raw_file.write_text(
            "\n".join(
                [
                    "0.00 1 2 3 4 5 6 7 8 1",
                    "bad_time 1 2 3 4 5 6 7 8 2",
                    "0.02 3 4 5 invalid 7 8 9 10 4",
                    "0.03 4 5 6 7 8 9 10 11 5",
                ]
            ),
            encoding="utf-8",
        )

        with caplog.at_level(logging.INFO):
            dataframe = parse_emg_file(raw_file)

        assert len(dataframe) == 2
        assert dataframe["label_id"].tolist() == [1, 5]
        assert dataframe["gesture"].tolist() == ["rest", "radial_deviation"]
        assert "Dropped 2 non-numeric rows" in caplog.text
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_parse_raw_dataset_drops_label_zero_by_default() -> None:
    workspace = _test_workspace()
    subject_dir = workspace / "EMG_data_for_gestures-master" / "2"
    subject_dir.mkdir(parents=True)
    raw_file = subject_dir / "recording_1.txt"
    try:
        raw_file.write_text(
            "\n".join(
                [
                    "0 0 0 0 0 0 0 0 0 0",
                    "1 1 1 1 1 1 1 1 1 7",
                ]
            ),
            encoding="utf-8",
        )

        result = parse_raw_dataset_with_report(
            workspace / "EMG_data_for_gestures-master"
        )

        assert result.files_failed == {}
        assert len(result.files_parsed) == 1
        assert result.dataframe["label_id"].tolist() == [7]
        assert result.dataframe["gesture"].tolist() == ["extended_palm"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_parse_raw_dataset_can_keep_label_zero() -> None:
    workspace = _test_workspace()
    subject_dir = workspace / "EMG_data_for_gestures-master" / "6"
    subject_dir.mkdir(parents=True)
    raw_file = subject_dir / "recording_keep_zero.txt"
    try:
        raw_file.write_text(
            "\n".join(
                [
                    "0 0 0 0 0 0 0 0 0 0",
                    "1 1 1 1 1 1 1 1 1 2",
                ]
            ),
            encoding="utf-8",
        )

        dataframe = parse_emg_file(raw_file, keep_unmarked=True)

        assert dataframe["label_id"].tolist() == [0, 2]
        assert dataframe["gesture"].tolist() == ["unmarked", "fist"]
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_summary_creation() -> None:
    workspace = _test_workspace()
    root = workspace / "EMG_data_for_gestures-master"
    subject_dir = root / "3"
    subject_dir.mkdir(parents=True)
    raw_file = subject_dir / "recording_2.txt"
    try:
        raw_file.write_text(
            "\n".join(
                [
                    "0 1 1 1 1 1 1 1 1 1",
                    "1 2 2 2 2 2 2 2 2 2",
                    "2 3 3 3 3 3 3 3 3 2",
                ]
            ),
            encoding="utf-8",
        )

        result = parse_raw_dataset_with_report(root)
        summary = create_dataset_summary(result)

        assert summary["total_rows"] == 3
        assert summary["number_of_subjects"] == 1
        assert summary["number_of_recordings"] == 1
        assert summary["label_distribution"] == {"1": 1, "2": 2}
        assert summary["rows_per_subject"] == {"3": 3}
        assert len(summary["files_parsed"]) == 1
        assert summary["files_failed"] == {}

        summary_path = workspace / "reports" / "metrics" / "dataset_summary.json"
        save_summary(summary, summary_path)

        assert json.loads(summary_path.read_text(encoding="utf-8"))["total_rows"] == 3
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
