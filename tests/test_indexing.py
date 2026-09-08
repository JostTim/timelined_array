import numpy as np
import pytest

from tests.conftest import AX0_SIZE_3D, AX1_SIZE_3D, AX2_SIZE_3D
from timelined_array import TimelinedArray
from timelined_array.time import BaseTimeArray


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


def test_normal_array_shape(standard_array_3D: np.ndarray):
    assert standard_array_3D.ndim == 3
    assert standard_array_3D.shape == (2, 4, 6)
    assert standard_array_3D[None].ndim == 4
    assert standard_array_3D[None].shape == (1, 2, 4, 6)
    assert not np.isscalar(slice(None))
    assert np.isscalar(15)

    test = np.array(5)
    assert test.shape == ()


def test_iteration(timelined_array_3D: TimelinedArray):
    # iteration should behave like a plain numpy array (iterate over first axis)
    items = list(timelined_array_3D)
    assert len(items) == AX0_SIZE_3D
    for item in items:
        assert item.shape == (AX1_SIZE_3D, AX2_SIZE_3D)


def test_scalar_indexing_first_axis(timelined_array_3D: TimelinedArray):
    # t[0] should index axis 0, which is not the time axis here
    item = timelined_array_3D[0]
    assert item.shape == (AX1_SIZE_3D, AX2_SIZE_3D)


def test_boolean_array_indexing(timelined_array_3D: TimelinedArray):
    # t[0] should index axis 0, which is not the time axis here
    index = np.array([False] * 50)
    trues = list(range(30, 36))
    index[trues] = True

    indexed_array = timelined_array_3D[:, index, :]
    timeline_should_be = timelined_array_3D.timeline[30:36]

    assert indexed_array.shape[1] == 6
    assert isinstance(indexed_array, BaseTimeArray)
    assert indexed_array.timeline.shape[0] == 6
    np.testing.assert_array_equal(
        indexed_array.timeline, np.array([timeline_should_be])
    )
