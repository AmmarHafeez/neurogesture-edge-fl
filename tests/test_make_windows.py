import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from src.data.make_windows import (
    create_window_summary,
    save_summary,
    save_windowed_dataset,
)
from src.preprocessing.windowing import (
    EMG_CHANNEL_COLUMNS,
    build_windows_from_dataframe,
)


def _test_workspace() -> Path:
    path = Path("tests/generated") / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def _rows(
    subject_id: str,
    recording_id: str,
    label_id: int,
    gesture: str,
    times: range,
) -> list[dict[str, object]]:
    rows = []
    for time in times:
        row = {
            "subject_id": subject_id,
            "recording_id": recording_id,
            "source_file": f"{subject_id}/{recording_id}.txt",
            "time": float(time),
            "label_id": label_id,
            "gesture": gesture,
        }
        row.update({channel: float(time) for channel in EMG_CHANNEL_COLUMNS})
        rows.append(row)
    return rows


def test_build_windows_shape_and_label_assignment() -> None:
    dataframe = pd.DataFrame(_rows("1", "rec_a", 2, "fist", range(6)))

    windowed = build_windows_from_dataframe(dataframe, window_size=3, stride=2)

    assert windowed.X.shape == (2, 3, 8)
    assert windowed.X.dtype == np.float32
    assert windowed.y.dtype == np.int64
    assert windowed.y.tolist() == [2, 2]
    assert windowed.gesture_names.tolist() == ["fist", "fist"]
    assert windowed.subject_ids.tolist() == ["1", "1"]
    assert windowed.recording_ids.tolist() == ["rec_a", "rec_a"]
    assert windowed.channel_names.tolist() == EMG_CHANNEL_COLUMNS


def test_mixed_label_group_does_not_produce_windows() -> None:
    rows = _rows("1", "rec_a", 2, "fist", range(3))
    rows += _rows("1", "rec_a", 3, "fist", range(3, 6))
    dataframe = pd.DataFrame(rows)

    windowed = build_windows_from_dataframe(dataframe, window_size=3, stride=1)

    assert windowed.X.shape == (0, 3, 8)
    assert windowed.y.tolist() == []


def test_windows_do_not_cross_recording_boundaries() -> None:
    rows = _rows("1", "rec_a", 1, "rest", range(2))
    rows += _rows("1", "rec_b", 1, "rest", range(2))
    dataframe = pd.DataFrame(rows)

    windowed = build_windows_from_dataframe(dataframe, window_size=3, stride=1)

    assert windowed.X.shape == (0, 3, 8)


def test_window_summary_generation_and_save() -> None:
    workspace = _test_workspace()
    try:
        rows = _rows("1", "rec_a", 1, "rest", range(5))
        rows += _rows("2", "rec_b", 2, "fist", range(5))
        dataframe = pd.DataFrame(rows)
        windowed = build_windows_from_dataframe(dataframe, window_size=3, stride=2)

        summary = create_window_summary(input_rows=len(dataframe), windowed_dataset=windowed)

        assert summary["input_rows"] == 10
        assert summary["window_size"] == 3
        assert summary["stride"] == 2
        assert summary["number_of_windows"] == 4
        assert summary["number_of_subjects"] == 2
        assert summary["number_of_recordings"] == 2
        assert summary["label_distribution_windows"] == {"1": 2, "2": 2}
        assert summary["gesture_distribution_windows"] == {"fist": 2, "rest": 2}
        assert summary["windows_per_subject"] == {"1": 2, "2": 2}
        assert summary["min_windows_per_subject"] == 2
        assert summary["max_windows_per_subject"] == 2
        assert summary["mean_windows_per_subject"] == 2.0

        summary_path = workspace / "reports" / "metrics" / "window_summary.json"
        save_summary(summary, summary_path)

        assert json.loads(summary_path.read_text(encoding="utf-8")) == summary

        windows_path = workspace / "data" / "processed" / "emg_windows.npz"
        save_windowed_dataset(windowed, windows_path)
        loaded = np.load(windows_path)

        assert loaded["X"].shape == (4, 3, 8)
        assert loaded["y"].tolist() == [1, 1, 2, 2]
        assert loaded["subject_ids"].tolist() == ["1", "1", "2", "2"]
        assert loaded["recording_ids"].tolist() == ["rec_a", "rec_a", "rec_b", "rec_b"]
        assert loaded["gesture_names"].tolist() == ["rest", "rest", "fist", "fist"]
        assert loaded["channel_names"].tolist() == EMG_CHANNEL_COLUMNS
        assert int(loaded["window_size"]) == 3
        assert int(loaded["stride"]) == 2
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
