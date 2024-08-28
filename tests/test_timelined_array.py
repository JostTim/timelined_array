import numpy as np
import pytest
from timelined_array import TimelinedArray, MaskedTimelinedArray
from timelined_array.time import (Timeline,
                                  StartBoundary,
                                  StoptBoundary,
                                  EdgePolicy,
                                  TimeIndexer,
                                  TimeMixin,
                                  TimePacker,
                                  Seconds)


@pytest.fixture
def timelined_array_1D():
    return TimelinedArray(np.random.rand(50),
                          timeline=np.arange(50),
                          time_dimension=0)


@pytest.fixture
def timelined_array_3D():
    return TimelinedArray(np.random.rand(25, 50, 75),
                          timeline=np.arange(50),
                          time_dimension=1)


def get_axis_parameters_3D():
    # affected_axis, time_dimension, shape
    # if None, we expect an AttributeError
    return [
        (0, 0, (50, 75)),
        (1, None, (25, 75)),
        (2, 1, (25, 50)),
        ((0, 2), 0, (50,)),
        (None, None, None)
    ]


@pytest.mark.parametrize("shape, timeline_dimension", [
    ((12,), 0),
    ((10, 30, 50), 2),
    ((29, 34, 61, 15, 4), 0)
])
def test_ta_array_shape(shape, timeline_dimension):
    ta_array = TimelinedArray(np.random.rand(*shape),
                              timeline=np.arange(
        shape[timeline_dimension]),
        time_dimension=timeline_dimension)

    assert ta_array.shape == shape
    assert ta_array.timeline.shape == (shape[timeline_dimension],)


def test_timeline_creation():
    timeline = Timeline(np.arange(10))
    assert isinstance(timeline, Timeline)
    assert timeline.max() == 9
    assert timeline.min() == 0


def test_start_boundary():
    assert StartBoundary.inclusive.value(5, 5)
    assert not StartBoundary.exclusive.value(5, 5)


def test_stop_boundary():
    assert StoptBoundary.inclusive.value(5, 5)
    assert not StoptBoundary.exclusive.value(5, 5)


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


def test_ta_swapaxes(timelined_array_3D):
    swapped = timelined_array_3D.swapaxes(1, 2)
    assert swapped.shape == (25, 75, 50)
    assert swapped.time_dimension == 2


@pytest.mark.parametrize("rolled_axis, end_position, expected_time_position, expected_shape", [
    (0, 2, 0, (50, 25, 75)),
    (2, 1, 2, (25, 75, 50)),
    (1, 3, 2, (25, 75, 50))
])
def test_ta_rollaxis(rolled_axis,
                     end_position,
                     expected_time_position,
                     expected_shape,
                     timelined_array_3D):
    # roll back the axis
    rolled = timelined_array_3D.rollaxis(rolled_axis, end_position)
    assert rolled.shape == expected_shape
    assert rolled.time_dimension == expected_time_position


def axis_affecting_functions_blueprint(timelined_array_3D,
                                       func,
                                       affected_axis,
                                       time_dimension,
                                       shape):
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
    axis_affecting_functions_blueprint(timelined_array_3D, np.mean,
                                       affected_axis, time_dimension, shape)


@pytest.mark.parametrize("affected_axis, time_dimension, shape", get_axis_parameters_3D())
def test_ta_sum(timelined_array_3D, affected_axis, time_dimension, shape):
    axis_affecting_functions_blueprint(timelined_array_3D, np.sum,
                                       affected_axis, time_dimension, shape)


@pytest.mark.parametrize("affected_axis, time_dimension, shape", get_axis_parameters_3D())
def test_ta_std(timelined_array_3D, affected_axis, time_dimension, shape):
    axis_affecting_functions_blueprint(timelined_array_3D, np.std,
                                       affected_axis, time_dimension, shape)


@pytest.mark.parametrize("affected_axis, time_dimension, shape", get_axis_parameters_3D())
def test_ta_var(timelined_array_3D, affected_axis, time_dimension, shape):
    axis_affecting_functions_blueprint(timelined_array_3D, np.var,
                                       affected_axis, time_dimension, shape)


def test_masked_ta_creation():
    data = np.random.rand(10, 10)
    mask = data > 0.5
    masked_ta = MaskedTimelinedArray(
        data, mask=mask, timeline=np.arange(10), time_dimension=0)
    assert isinstance(masked_ta, MaskedTimelinedArray)
    assert masked_ta.shape == (10, 10)
    assert masked_ta.time_dimension == 0


def test_seconds_to_index():
    sec = Seconds(10)
    assert sec.to_index(2) == 20
