import numpy as np
import pytest

from tests.conftest import ARRAY_SHAPE_1D, TIME_AXIS_1D
from timelined_array import TimelinedArray
from timelined_array.time import (
    EndEdgePolicy,
    StartEdgePolicy,
    TimeIndexer,
)


def test_timelined_array_1D_properties(timelined_array_1D: TimelinedArray):
    assert timelined_array_1D.time_dimension == TIME_AXIS_1D
    assert timelined_array_1D.timeline.shape == ARRAY_SHAPE_1D


def test_timelined_array_indexing(timelined_array_1D: TimelinedArray):
    sub_array = timelined_array_1D[10:20]
    assert isinstance(sub_array, TimelinedArray)
    assert sub_array.shape == (10,)
    assert np.array_equal(sub_array.timeline, np.arange(10, 20))
    assert timelined_array_1D[[1]] == timelined_array_1D[1]


@pytest.mark.parametrize(
    "shape, timeline_dimension",
    [((12,), 0), ((10, 30, 50), 2), ((29, 34, 61, 15, 4), 0)],
)
def test_ta_array_shape(shape: tuple[int, ...], timeline_dimension: int):
    ta_array = TimelinedArray(
        np.random.rand(*shape),
        timeline=np.arange(shape[timeline_dimension]),
        time_dimension=timeline_dimension,
    )

    assert ta_array.shape == shape
    assert ta_array.timeline.shape == (shape[timeline_dimension],)


def test_timelined_array_creation(timelined_array_1D: TimelinedArray):

    with pytest.raises(ValueError):
        # when converting the timelined_array_1D, wich is a valid timeline array, to np.array,
        # we strip out the timeline and time dimension informations, so we should supply it when
        # creating a new timelined array, or it will fail. So here, it fails as we do not supply these
        TimelinedArray(np.array(timelined_array_1D))

    with pytest.raises(ValueError):
        # Time dimension should be an int, expected to raise
        TimelinedArray(np.array(timelined_array_1D), time_dimension=5.5)  # type: ignore


def test_start_edge_policy():

    for inclusive, exclusive in [
        (StartEdgePolicy[inc_str].value, StartEdgePolicy[exc_str].value)
        for inc_str, exc_str in [("inclusive", "exclusive"), ("inc", "exc")]
    ]:
        assert inclusive(5, 5)  # 5 is higher or equal to 5
        assert not exclusive(5, 5)  # 5 is NOT strictly higher than 5

        assert inclusive(6, 5)  # 6 is higher or equal to 5
        assert exclusive(6, 5)  # 6 is strictly higher than 5

        assert not inclusive(4, 5)  # 4 is NOT higher or equal to 5
        assert not exclusive(4, 5)  # 4 is NOT strictly higher than 5


def test_end_edge_policy():

    for inclusive, exclusive in [
        (EndEdgePolicy[inc_str].value, EndEdgePolicy[exc_str].value)
        for inc_str, exc_str in [("inclusive", "exclusive"), ("inc", "exc")]
    ]:
        assert inclusive(5, 5)  # 5 is lower or equal to 5
        assert not exclusive(5, 5)  # 5 is NOT strictly lower than 5

        assert inclusive(4, 5)  # 4 is lower or equal to 5
        assert exclusive(4, 5)  # 4 is strictly lower than 5

        assert not inclusive(6, 5)  # 6 is NOT lower or equal to 5
        assert not exclusive(6, 5)  # 6 is NOT strictly lower than 5


def test_time_indexer(timelined_array_1D: TimelinedArray):
    indexer = TimeIndexer(timelined_array_1D)
    assert indexer.time_to_index(5) == 5
    assert indexer.time_to_index(slice(2, 5)) == slice(2, 5, 1)


def test_time_mixin_methods(timelined_array_1D: TimelinedArray):
    assert timelined_array_1D.max_time() == 49
    assert timelined_array_1D.min_time() == 0


# def test_seconds_to_index():
#     time_unit = TimeUnit(10)
#     assert time_unit.to_index(2) == 20


def test_rebase_timelined_array(timelined_array_1D: TimelinedArray):

    array = timelined_array_1D.rebase_timeline(at=16)
    assert array.timeline[0] == 16


def test_offset_timeline(timelined_array_1D: TimelinedArray):
    array = timelined_array_1D.offset_timeline(offset=84)
    assert array.timeline[22] == (22 + 84)


# def test_shift_period(timelined_array_1D: TimelinedArray):

#     with pytest.raises(NotImplementedError):
#         timelined_array_1D.shift_values(period=slice(8, 10))


def test_pack(timelined_array_1D: TimelinedArray):

    timeline, array = timelined_array_1D.pack
    assert timeline is timelined_array_1D.timeline
    assert np.all(array == timelined_array_1D)
