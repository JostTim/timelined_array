from functools import partial
from types import GeneratorType
from typing import Protocol, cast

import numpy as np
import pytest

from tests.conftest import (
    AX0_SIZE_3D,
    AX1_SIZE_3D,
    AX2_SIZE_3D,
    TIME_AXIS_3D,
    TIME_AXIS_SIZE_3D,
)
from timelined_array import TimelinedArray
from timelined_array.time import BaseTimeArray


class NumpyAggFunction(Protocol):
    def __call__(
        self, axis: int | tuple[int, ...] | None = None
    ) -> TimelinedArray | np.ndarray | float: ...


def test_ta3D_basic_attributes(timelined_array_3D: TimelinedArray):
    assert timelined_array_3D.time_dimension == TIME_AXIS_3D
    assert timelined_array_3D.timeline.shape == (TIME_AXIS_SIZE_3D,)


def get_axis_parameters_3D():
    # affected_axis, time_dimension, shape
    # if None, we affect all axes
    return [0, 1, 2, (0, 2), (0, 1), (1, 2), (0, 1, 2), None]


def yield_numpy_function_alternatives(
    array: TimelinedArray, function_name: str
) -> GeneratorType[NumpyAggFunction]:
    yield partial(getattr(np, function_name), array)
    yield getattr(array, function_name)


def parametrizeable_dimension_sanity_checker(
    timelined_array_3D: TimelinedArray,
    function_name: str,
    affected_axis: int | tuple[int, ...] | None,
):
    initial_time_axis = timelined_array_3D.time_dimension
    initial_shape = timelined_array_3D.shape
    initial_axes = tuple(range(timelined_array_3D.ndim))
    for function in yield_numpy_function_alternatives(
        timelined_array_3D, function_name
    ):
        if affected_axis is None:
            affected_axes = None
        elif not isinstance(affected_axis, tuple):
            affected_axes = (affected_axis,)
        else:
            affected_axes = affected_axis

        result = function(axis=affected_axis)
        if affected_axes is None or sorted(affected_axes) == sorted(initial_axes):
            # if all dimensions collapsed, the return should be a scalar
            assert not isinstance(result, np.ndarray)
            # as the results is not a numpy array anymore, there is no more to check
            continue
        elif initial_time_axis in affected_axes:
            # if the time dimensions collapsed (but not all of the dimensions), the return should be a normal numpy array
            assert isinstance(result, np.ndarray)
            assert not hasattr(result, "time_dimension")
        else:
            # else, the result should still be a TimelinedArray, and we check the new time_axis
            new_time_dimension = initial_time_axis - len(
                [axis for axis in affected_axes if axis < initial_time_axis]
            )
            assert cast(TimelinedArray, result).time_dimension == new_time_dimension

        result = cast(TimelinedArray | np.ndarray, result)

        new_shape = tuple(
            [
                axis_size
                for index, axis_size in enumerate(initial_shape)
                if index not in affected_axes
            ]
        )

        assert result.shape == new_shape


@pytest.mark.parametrize("affected_axis", get_axis_parameters_3D())
def test_ta3D_mean_parametrized(
    timelined_array_3D: TimelinedArray,
    affected_axis: int | tuple[int, ...] | None,
):
    parametrizeable_dimension_sanity_checker(timelined_array_3D, "mean", affected_axis)


@pytest.mark.parametrize("affected_axis", get_axis_parameters_3D())
def test_ta3D_sum_parametrized(
    timelined_array_3D: TimelinedArray, affected_axis: int | tuple[int, ...] | None
):
    parametrizeable_dimension_sanity_checker(timelined_array_3D, "sum", affected_axis)


@pytest.mark.parametrize("affected_axis", get_axis_parameters_3D())
def test_ta3D_std_parametrized(
    timelined_array_3D: TimelinedArray, affected_axis: int | tuple[int, ...] | None
):
    parametrizeable_dimension_sanity_checker(timelined_array_3D, "std", affected_axis)


@pytest.mark.parametrize("affected_axis", get_axis_parameters_3D())
def test_ta3D_var_parametrized(
    timelined_array_3D: TimelinedArray, affected_axis: int | tuple[int, ...] | None
):
    parametrizeable_dimension_sanity_checker(timelined_array_3D, "var", affected_axis)


@pytest.mark.parametrize(
    "rolled_axis, end_position, expected_time_position, expected_shape",
    [
        (0, 2, 0, (AX1_SIZE_3D, AX0_SIZE_3D, AX2_SIZE_3D)),
        (2, 1, 2, (AX0_SIZE_3D, AX2_SIZE_3D, AX1_SIZE_3D)),
        (1, 3, 2, (AX0_SIZE_3D, AX2_SIZE_3D, AX1_SIZE_3D)),
    ],
)
def test_ta3D_rollaxis_parametrized(
    rolled_axis: int,
    end_position: int,
    expected_time_position: int,
    expected_shape: tuple[int, ...],
    timelined_array_3D: TimelinedArray,
):
    # roll back the axis
    rolled = timelined_array_3D.rollaxis(rolled_axis, end_position)

    assert rolled.shape == expected_shape
    assert rolled.time_dimension == expected_time_position


def test_ta3D_moveaxis(timelined_array_3D: TimelinedArray):

    array = timelined_array_3D.moveaxis(TIME_AXIS_3D, 2)
    assert array.time_dimension == 2
    assert array.shape[2] == TIME_AXIS_SIZE_3D
    assert array.shape == (AX0_SIZE_3D, AX2_SIZE_3D, TIME_AXIS_SIZE_3D)

    array = timelined_array_3D.moveaxis(0, 2)
    assert array.time_dimension == 0
    assert array.shape == (AX1_SIZE_3D, AX2_SIZE_3D, AX0_SIZE_3D)


def test_ta3D_transpose(timelined_array_3D: TimelinedArray):
    transposed = timelined_array_3D.transpose(2, 0, 1)
    assert transposed.shape == (AX2_SIZE_3D, AX0_SIZE_3D, AX1_SIZE_3D)
    assert transposed.time_dimension == 2
    assert isinstance(transposed, BaseTimeArray)

    transposed = timelined_array_3D.T
    assert transposed.shape == (AX2_SIZE_3D, AX1_SIZE_3D, AX0_SIZE_3D)
    assert transposed.time_dimension == 1
    assert isinstance(transposed, BaseTimeArray)


def test_ta3D_swapaxes(timelined_array_3D: TimelinedArray):
    swapped = timelined_array_3D.swapaxes(TIME_AXIS_3D, 2)
    assert swapped.shape == (AX0_SIZE_3D, AX2_SIZE_3D, AX1_SIZE_3D)
    assert swapped.time_dimension == 2


def test_timelined_array_from_iterable(timelined_array_3D: TimelinedArray):
    assert timelined_array_3D.timeline.min() == 0
    assert timelined_array_3D.timeline.max() == TIME_AXIS_SIZE_3D - 1

    array = TimelinedArray.align_from_iterable(timelined_array_3D)  # ty: ignore[invalid-argument-type]
    assert np.all(array == timelined_array_3D)
