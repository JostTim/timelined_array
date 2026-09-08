import numpy as np
import pytest

from tests.conftest import AX0_SIZE_3D, AX1_SIZE_3D, AX2_SIZE_3D, TIME_AXIS_SIZE_1D
from timelined_array import BaseTimeArray, Timeline, TimelinedArray


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


def test_scalar_indexing_3D(timelined_array_3D: TimelinedArray):
    normal_array = timelined_array_3D.__array__()

    t_result = timelined_array_3D[0, 5, 0]
    n_result = normal_array[0, 5, 0]
    assert isinstance(t_result, float)
    assert t_result == n_result


def test_iteration_3D(timelined_array_3D: TimelinedArray):
    # iteration should behave like a plain numpy array (iterate over first axis)
    assert len(timelined_array_3D) == AX0_SIZE_3D
    for item in timelined_array_3D:
        assert item.shape == (AX1_SIZE_3D, AX2_SIZE_3D)  # ty: ignore[unresolved-attribute]
        assert isinstance(item, TimelinedArray)
        assert item.time_dimension == 0
        np.testing.assert_array_equal(item.timeline, timelined_array_3D.timeline)
        for sub_item in item:
            assert sub_item.shape == (AX2_SIZE_3D,)  # ty: ignore[unresolved-attribute]
            assert not isinstance(sub_item, TimelinedArray)
            assert not hasattr(sub_item, "timeline")


def test_iteration_1D(timelined_array_1D: TimelinedArray):
    assert len(timelined_array_1D) == TIME_AXIS_SIZE_1D
    for item in timelined_array_1D:
        assert isinstance(item, float)


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
    assert isinstance(timeline_should_be, Timeline)
    np.testing.assert_array_equal(indexed_array.timeline, timeline_should_be)


def test_sorting_array_indexing(timelined_array_3D: TimelinedArray):

    sort_index = [12, 34, 45, 34]

    indexed_array = timelined_array_3D[:, sort_index, :]
    assert indexed_array.timeline.tolist() == sort_index  # ty: ignore[unresolved-attribute]
    normal_indexed_array = timelined_array_3D.__array__()[:, sort_index, :]
    np.testing.assert_array_equal(indexed_array[0, :, 0], normal_indexed_array[0, :, 0])
