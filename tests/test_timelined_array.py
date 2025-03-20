import numpy as np
import pytest
from timelined_array import TimelinedArray, MaskedTimelinedArray
from timelined_array.time import (
    Timeline,
    StartBoundary,
    StoptBoundary,
    EdgePolicy,
    TimeIndexer,
    TimeMixin,
    TimePacker,
    Seconds,
)
import pickle
from pathlib import Path


@pytest.fixture
def timelined_array_1D():
    return TimelinedArray(np.random.rand(50), timeline=np.arange(50), time_dimension=0)


@pytest.fixture
def timelined_array_3D():
    return TimelinedArray(np.random.rand(25, 50, 75), timeline=np.arange(50), time_dimension=1)


@pytest.fixture
def masked_timelined_array_1D():
    data = np.random.rand(50)
    mask = data > 0.5
    return MaskedTimelinedArray(data, mask=mask, timeline=np.arange(50), time_dimension=0)


@pytest.fixture
def pickle_path():
    file_path = Path("serialized_test_3D_ta_array.pickle")
    yield file_path
    if file_path.exists():
        file_path.unlink(missing_ok=True)


def test_timelined_array_1D_properties(timelined_array_1D):
    assert timelined_array_1D.time_dimension == 0
    assert timelined_array_1D.timeline.shape == (50,)


def test_timelined_array_3D_properties(timelined_array_3D):
    assert timelined_array_3D.time_dimension == 1
    assert timelined_array_3D.timeline.shape == (50,)


def test_masked_timelined_array_1D_properties(masked_timelined_array_1D):
    assert masked_timelined_array_1D.time_dimension == 0
    assert np.all(masked_timelined_array_1D.mask == (masked_timelined_array_1D.data > 0.5))
    assert masked_timelined_array_1D.timeline.shape == (50,)


def test_timelined_array_indexing(timelined_array_1D):
    sub_array = timelined_array_1D[10:20]
    assert isinstance(sub_array, TimelinedArray)
    assert sub_array.shape == (10,)
    assert np.array_equal(sub_array.timeline, np.arange(10, 20))
    assert timelined_array_1D[[1]] == timelined_array_1D[1]


def test_masked_timelined_array_indexing(masked_timelined_array_1D):
    sub_array = masked_timelined_array_1D[10:20]
    assert isinstance(sub_array, MaskedTimelinedArray)
    assert sub_array.shape == (10,)
    assert np.array_equal(sub_array.timeline, np.arange(10, 20))
    assert np.all(sub_array.mask == (sub_array.data > 0.5))


def get_axis_parameters_3D():
    # affected_axis, time_dimension, shape
    # if None, we expect an AttributeError
    return [(0, 0, (50, 75)), (1, None, (25, 75)), (2, 1, (25, 50)), ((0, 2), 0, (50,)), (None, None, None)]


@pytest.mark.parametrize("shape, timeline_dimension", [((12,), 0), ((10, 30, 50), 2), ((29, 34, 61, 15, 4), 0)])
def test_ta_array_shape(shape, timeline_dimension):
    ta_array = TimelinedArray(
        np.random.rand(*shape), timeline=np.arange(shape[timeline_dimension]), time_dimension=timeline_dimension
    )

    assert ta_array.shape == shape
    assert ta_array.timeline.shape == (shape[timeline_dimension],)


def test_timeline_creation():
    timeline = Timeline(np.arange(10))
    assert isinstance(timeline, Timeline)
    assert timeline.max() == 9
    assert timeline.min() == 0
    assert 5 in timeline
    assert timeline.step == 1

    with pytest.raises(NotImplementedError):
        timeline = Timeline(np.random.rand(10), uniform_space=True)
    # assert np.unique(np.diff(timeline)).size == 1


def test_timelined_array_creation(timelined_array_1D):

    with pytest.raises(ValueError):
        array = TimelinedArray(np.array(timelined_array_1D))

    with pytest.raises(ValueError):
        array = TimelinedArray(np.array(timelined_array_1D), time_dimension=5.5)


def test_pickle_unpickle(timelined_array_3D, pickle_path):
    # Pickle the numpy array
    with open(pickle_path, "wb") as f:
        pickle.dump(timelined_array_3D, f)

    # Unpickle the numpy array
    with open(pickle_path, "rb") as f:
        unserialized_3D_array = pickle.load(f)

    # Assert that the original and unpickled arrays are the same
    np.testing.assert_array_equal(timelined_array_3D, unserialized_3D_array)
    np.testing.assert_array_equal(timelined_array_3D.timeline, unserialized_3D_array.timeline)
    assert timelined_array_3D.time_dimension == unserialized_3D_array.time_dimension


def test_start_boundary():
    assert StartBoundary.inclusive.value(5, 5)  # 5 is => than 5
    assert not StartBoundary.exclusive.value(5, 5)  # 5 is not > than 5


def test_stop_boundary():
    assert StoptBoundary.inclusive.value(5, 5)  # 5 is =< than 5
    assert not StoptBoundary.exclusive.value(5, 5)  # 5 is not < than 5


def test_edge_policy():
    assert EdgePolicy.start.value == StartBoundary
    assert EdgePolicy.stop.value == StoptBoundary


def test_time_indexer(timelined_array_1D):
    indexer = TimeIndexer(timelined_array_1D)
    assert indexer.time_to_index(5) == 5
    assert indexer.time_to_index(slice(2, 5)) == slice(2, 5, 1)


def test_time_mixin_methods(timelined_array_1D):
    assert timelined_array_1D.sec_max() == 49
    assert timelined_array_1D.sec_min() == 0


def test_ta_transpose(timelined_array_3D):
    transposed = timelined_array_3D.transpose(2, 0, 1)
    assert transposed.shape == (75, 25, 50)
    assert transposed.time_dimension == 2

    transposed = timelined_array_3D.T
    assert transposed.shape == (75, 50, 25)
    assert transposed.time_dimension == 1


def test_ta_swapaxes(timelined_array_3D):
    swapped = timelined_array_3D.swapaxes(1, 2)
    assert swapped.shape == (25, 75, 50)
    assert swapped.time_dimension == 2


@pytest.mark.parametrize(
    "rolled_axis, end_position, expected_time_position, expected_shape",
    [(0, 2, 0, (50, 25, 75)), (2, 1, 2, (25, 75, 50)), (1, 3, 2, (25, 75, 50))],
)
def test_ta_rollaxis(rolled_axis, end_position, expected_time_position, expected_shape, timelined_array_3D):
    # roll back the axis
    rolled = timelined_array_3D.rollaxis(rolled_axis, end_position)
    assert rolled.shape == expected_shape
    assert rolled.time_dimension == expected_time_position


def axis_affecting_functions_blueprint(timelined_array_3D, func, affected_axis, time_dimension, shape):
    result = func(timelined_array_3D, axis=affected_axis)
    if time_dimension is None:
        with pytest.raises(AttributeError):
            result.time_dimension
    else:
        assert result.time_dimension == time_dimension

    if shape is None:
        with pytest.raises(AttributeError):
            result.shape
    else:
        assert result.shape == shape


@pytest.mark.parametrize("affected_axis, time_dimension, shape", get_axis_parameters_3D())
def test_ta_mean(timelined_array_3D, affected_axis, time_dimension, shape):
    axis_affecting_functions_blueprint(timelined_array_3D, np.mean, affected_axis, time_dimension, shape)


@pytest.mark.parametrize("affected_axis, time_dimension, shape", get_axis_parameters_3D())
def test_ta_sum(timelined_array_3D, affected_axis, time_dimension, shape):
    axis_affecting_functions_blueprint(timelined_array_3D, np.sum, affected_axis, time_dimension, shape)


@pytest.mark.parametrize("affected_axis, time_dimension, shape", get_axis_parameters_3D())
def test_ta_std(timelined_array_3D, affected_axis, time_dimension, shape):
    axis_affecting_functions_blueprint(timelined_array_3D, np.std, affected_axis, time_dimension, shape)


@pytest.mark.parametrize("affected_axis, time_dimension, shape", get_axis_parameters_3D())
def test_ta_var(timelined_array_3D, affected_axis, time_dimension, shape):
    axis_affecting_functions_blueprint(timelined_array_3D, np.var, affected_axis, time_dimension, shape)


def test_masked_ta_creation():
    data = np.random.rand(10, 10)
    mask = data > 0.5
    masked_ta = MaskedTimelinedArray(data, mask=mask, timeline=np.arange(10), time_dimension=0)
    assert isinstance(masked_ta, MaskedTimelinedArray)
    assert masked_ta.shape == (10, 10)
    assert masked_ta.time_dimension == 0


def test_seconds_to_index():
    sec = Seconds(10)
    assert sec.to_index(2) == 20


def test_timelined_array_from_iterable(timelined_array_3D):
    array = TimelinedArray.align_from_iterable(timelined_array_3D)
    assert np.all(array == timelined_array_3D)


def test_rebase_timelined_array(timelined_array_1D):

    array = timelined_array_1D.rebase_timeline(at=16)
    assert array.timeline[16] == 0


def test_offset_timeline(timelined_array_1D):
    array = timelined_array_1D.offset_timeline(offset=84)
    assert array.timeline[22] == (22 + 84)


def test_shift_period(timelined_array_1D):

    with pytest.raises(NotImplementedError):
        array = timelined_array_1D.shift_values(period=slice(8, 10))


def test_moveaxis(timelined_array_3D):

    array = timelined_array_3D.moveaxis(1, 2)
    assert array.time_dimension == 2
    assert array.shape[2] == 50

    array = timelined_array_3D.moveaxis(0, 2)
    assert array.time_dimension == 0
    assert array.shape[1] == 75


def test_pack(timelined_array_1D):

    timeline, array = timelined_array_1D.pack
    assert timeline is timelined_array_1D.timeline
    assert np.all(array == timelined_array_1D)
