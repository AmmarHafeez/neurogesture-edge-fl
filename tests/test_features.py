import numpy as np
import pytest

from src.preprocessing.features import mean_absolute_value


def test_mean_absolute_value_per_channel() -> None:
    window = np.array([[-1.0, 2.0], [3.0, -4.0]])

    features = mean_absolute_value(window)

    np.testing.assert_allclose(features, np.array([2.0, 3.0]))


def test_mean_absolute_value_requires_2d_input() -> None:
    with pytest.raises(ValueError):
        mean_absolute_value(np.array([1.0, 2.0, 3.0]))
