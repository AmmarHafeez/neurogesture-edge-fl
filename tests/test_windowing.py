import numpy as np
import pytest

from src.preprocessing.windowing import sliding_windows


def test_sliding_windows_shape() -> None:
    array = np.arange(20).reshape(10, 2)

    windows = sliding_windows(array, window_size=4, step_size=2)

    assert windows.shape == (4, 4, 2)
    np.testing.assert_array_equal(windows[0], array[:4])


def test_sliding_windows_rejects_non_positive_sizes() -> None:
    array = np.arange(20).reshape(10, 2)

    with pytest.raises(ValueError):
        sliding_windows(array, window_size=0, step_size=2)
