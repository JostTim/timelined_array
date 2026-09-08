import numpy as np
import pytest


@pytest.fixture
def standard_array_3D():
    return np.array(
        [
            [
                [1, 2, 3, 4, 5, 6],
                [0, 3, 6, 9, 12, 15],
                [1, 3, 9, 27, 81, 243],
                [1, 5, 10, 15, 20, 25],
            ],
            [
                [2, 4, 6, 8, 10, 12],
                [1, 3, 5, 7, 9, 11],
                [1, 4, 9, 16, 25, 36],
                [-1, -4, -8, -12, -16, -18],
            ],
        ]
    )


def test_array_shape(standard_array_3D: np.ndarray):
    assert standard_array_3D.ndim == 3
    assert standard_array_3D.shape == (2, 4, 6)
    assert standard_array_3D[None].ndim == 4
    assert standard_array_3D[None].shape == (1, 2, 4, 6)
    assert not np.isscalar(slice(None))
    assert np.isscalar(15)

    test = np.array(5)
    assert test.shape == ()
