# -*- coding: utf-8 -*-

import numpy as np
from logging import getLogger
from enum import Enum
import operator

from numpy.typing import NDArray
from types import MethodType
from typing import Tuple, List, Protocol, Type, Callable, Any, Optional

OperatorType = Callable[[Any, Any], bool]

# class syntax

logger = getLogger("timelined_array")


class TimeCompatibleProtocol(Protocol):

    time_dimension: int
    timeline: "Timeline"

    def __getitem__(self, index) -> np.ndarray: ...

    def __array__(self) -> np.ndarray: ...

    @property
    def shape(self) -> Tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def itime(self) -> "TimeIndexer": ...

    def _get_array_cls(self) -> "Type": ...

    def transpose(self): ...


class Timeline(np.ndarray):
    _step = None
    _max_step = None
    max_step_mult = 2

    def __new__(cls, input_array, uniform_space=False):
        """Create a new instance of the Timeline class.

        Args:
            input_array: The input array to create the Timeline object from.
            uniform_space (bool): Flag to indicate if a uniformly spaced timeline is desired.

        Returns:
            Timeline: A new instance of the Timeline class.
        """

        if uniform_space:
            # if we want a uniformly spaced timeline from start to stop of the current timeline.
            obj = Timeline._uniformize(input_array)
        else:
            if isinstance(input_array, Timeline):
                return input_array
            obj = np.asarray(input_array).view(cls)

        return obj

    def __array_finalize__(self, obj):
        """Finalize the array when subclassing a numpy array.

        Args:
            self: The subclassed array.
            obj: The original array object being subclassed.

        Returns:
            None
        """

        pass

    def __setstate__(self, state):
        """Set the state of the object.

        Args:
            state: The state to set for the object.
        """

        # try:
        #     super().__setstate__(state[0:-2])  # old deserializer
        # except TypeError:
        super().__setstate__(state)  # new one

    def __contains__(self, time_value):
        """Check if the time_value is within the range of the TimeRange object."""

        return self.min() <= time_value <= self.max()

    def max(self):
        """Return the maximum value in the iterable."""

        return super().max().item()

    def min(self):
        """Return the minimum value in the tensor."""

        return super().min().item()

    @classmethod
    def _uniformize(cls, timeline):
        """Uniformize the given timeline data.

        Args:
            cls: The class instance.
            timeline: The timeline data to be uniformized.

        Raises:
            NotImplementedError: This function is not yet implemented.

        Returns:
            None
        """

        raise NotImplementedError("Upcoming function")
        # obj = np.linspace(input_array[0], input_array[1], len(input_array)).view(cls)
        # TODO : do numpy.interp(np.arange(0, len(a), 1.5), np.arange(0, len(a)), a)
        # interp to get a fixed number of points ?

    def uniformize(self):
        """Uniformize the elements of the list using the _uniformize method."""

        self[:] = self._uniformize(self)

    @property
    def step(self):
        """Mean time between two timeline points. Must be strictly decreasing or increasing to be calculated"""
        if self._step is None:
            diff = np.diff(self)
            # make sure it is continuously rising or decreasing
            if np.all(diff >= 0) or np.all(diff <= 0):
                self._step = np.mean(diff)
            else:
                raise ValueError(
                    "Cannot determine the step value of the timeline. "
                    "It must be strictly increasing or strictly decreasing."
                )
        return self._step

    @property
    def max_step(self):
        """Largest time between two timeline points, multiplied by max_step_mult"""
        if self._max_step is None:
            diff = np.diff(self)
            self._max_step = diff[np.argmax(np.absolute(diff))]
        return self._max_step * self.max_step_mult


class StartBoundary(Enum):
    inclusive = operator.ge
    exclusive = operator.gt
    inc = operator.ge
    exc = operator.gt


class StoptBoundary(Enum):
    inclusive = operator.le
    exclusive = operator.lt
    inc = operator.le
    exc = operator.lt


class EdgePolicy(Enum):
    start = StartBoundary
    stop = StoptBoundary


class TimeIndexer:
    """The time indexer indexes by default from >= to the time start, and strictly < to time stop"""

    _start_operation: OperatorType
    _stop_operation: OperatorType

    def __init__(self, array: "BaseTimeArray", start="inclusive", stop="exclusive"):
        self.array = array
        self.set_edge_policy(start, stop)

    def set_edge_policy(self, start="inclusive", stop="exclusive"):
        self._start_operation = EdgePolicy["start"].value[start].value
        self._stop_operation = EdgePolicy["stop"].value[stop].value
        return self

    def time_to_index(
        self, time: float | int | slice | Tuple[int | float] | List[float | int | slice | Tuple[int | float]]
    ):
        """Converts time to index based on different input types.

        Args:
            time (float | int | slice | Tuple[int | float] | List[float | int | slice | Tuple[int | float]]):
                The time value or range to be converted to index.

        Returns:
            int: The index corresponding to the input time value or range.

        Raises:
            ValueError: If the input time type is not supported.
        """
        # argument index may be a slice or a scalar. Units of index should be in second. Returns a slice as index
        # this is sort of a wrapper for get_iindex that does the heavy lifting.
        # this function just makes sure to pass arguments to it corectly depending
        # on if the time index is a single value or a slice.

        if isinstance(time, slice):
            return self.get_iindex(time.start, time.stop, time.step)
        elif isinstance(time, list):
            return np.array([self.time_to_index(t) for t in time])
        elif isinstance(time, tuple):
            return self.get_iindex(*[time[i] if len(time) > i else None for i in range(3)])
        elif isinstance(time, (int, float)):
            return self.get_iindex(sec_start=time).start
        else:
            raise ValueError("Cannot process time to index")

    seconds_to_index = time_to_index

    def _insert_time_index(self, time_index):
        """Inserts a time index into the full index.

        Args:
            time_index: The index to be inserted into the full index.

        Returns:
            tuple: The full index with the time index inserted.
        """
        # put the integer value at the position of time index at the right position
        # (time_dimension) in the tuple of all sliced dimensions

        full_index = [slice(None)] * len(self.array.shape)
        full_index[self.array.time_dimension] = time_index

        return tuple(full_index)

    def __getitem__(self, index) -> "TimelinedArray | MaskedTimelinedArray | np.ndarray":
        """Get item from TimelinedArray, MaskedTimelinedArray, or np.ndarray based on the given index.

        Args:
            index: int, float, slice, np.integer, np.floating
                The index to retrieve the item from the array.

        Returns:
            TimelinedArray | MaskedTimelinedArray | np.ndarray
                The item at the specified index.

        Raises:
            ValueError: If the index is iterable and not a valid type for indexing on the time dimension.
        """

        if hasattr(index, "__iter__"):
            # if not isinstance(index,(int,float,slice,np.integer,np.floating)):
            raise ValueError(
                "Isec allow only indexing on time dimension. Index must be either int, float or slice, not iterable"
            )

        iindex_time = self.time_to_index(index)
        full_iindex = self._insert_time_index(iindex_time)
        # print("new full index : ",iindex_time)
        logger.debug(f"About to index over time with iindex_time {iindex_time} and full_iindex {full_iindex}")
        return self.array[full_iindex]

    def get_iindex(self, sec_start=None, sec_stop=None, sec_step=None):
        """Get the index range based on the given start, stop, and step values in seconds.

        Args:
            sec_start (float): The start time in seconds. If None, start index will be 0.
            sec_stop (float): The stop time in seconds. If None, stop index will be the length of the timeline.
            sec_step (float): The step size in seconds. If None, step size will be 1.

        Returns:
            slice: A slice object representing the index range based on the given start, stop, and step values.
        """
        # converts a time index (follows a slice syntax, but in time units) to integer units

        timeline_max_step = abs(self.array.timeline.max_step)

        if sec_start is None:
            start = 0
        else:
            if self._start_operation(sec_start, self.array.timeline[0]):
                start = np.argmax(self.array.timeline >= sec_start)
            else:
                start = 0

            if abs(self.array.timeline[start] - sec_start) > timeline_max_step:
                raise IndexError(
                    f"The start time value {sec_start} you searched for is not in the timeline of this array "
                    f"(timeline starts at {self.array.timeline[0]}, allowed jitter = {timeline_max_step} :"
                    " +/- 2 times the max step between two timeline points"
                )

        if sec_stop is None:
            stop = len(self.array.timeline)
        # elif sec_stop < 0 : Here we allowed for negative indexing but as timeline can have negative values
        # , i removed this posibility
        #    stop = np.argmin(self.array.timeline<self.array.timeline[-1]+sec_stop)
        else:
            if self._stop_operation(sec_stop, self.array.timeline[-1]):
                stop = np.argmin(self.array.timeline < sec_stop)
            else:
                stop = len(self.array.timeline) - 1

            if abs(self.array.timeline[stop] - sec_stop) > timeline_max_step:
                raise IndexError(
                    f"The end time value {sec_stop} you searched for is not in the timeline of this array "
                    f"(timeline ends at {self.array.timeline[-1]} , allowed jitter = {timeline_max_step} : "
                    "+/- 2 times the max step between two timeline points"
                )

        if sec_step is None:
            step = 1
        else:
            step = int(np.round(sec_step / self.array.timeline.step))
            if step < 1:
                step = 1
        return slice(start, stop, step)

    def __call__(self, start="inclusive", stop="exclusive"):
        return self.set_edge_policy(start, stop)


class TimeMixin:

    time_dimension: int
    timeline: Timeline
    start_policy = "inclusive"
    stop_policy = "exclusive"

    # ndarray inherited
    ndim: int
    shape: Tuple[int, ...]
    __sub__: Callable

    def _time_dimension_in_axis(self, axis: int | Tuple[int, ...] | None) -> bool:
        """Check if the time dimension is present in the specified axis.

        Args:
            axis (int | Tuple[int, ...] | None): The axis to check for the time dimension.

        Returns:
            bool: True if the time dimension is present in the axis, False otherwise.
        """

        if (
            axis is None
            or axis == self.time_dimension
            or (isinstance(axis, (list, tuple)) and self.time_dimension in axis)
        ):
            return True
        return False

    def _get_time_dimension_after_axis_removal(self, axis_removed) -> int:
        """Return the time dimension after removing specified axis.

        Args:
            axis_removed (int or tuple): The axis or axes to be removed.

        Returns:
            int: The time dimension after removing the specified axis or axes.

        Raises:
            ValueError: If the time dimension would be discarded after axis removal.
        """

        if not isinstance(axis_removed, tuple):
            axis_removed = (axis_removed,)
        axis_removed = sorted(axis_removed)

        final_time_dimension = self.time_dimension
        for axis in axis_removed:
            if axis < self.time_dimension:
                final_time_dimension -= 1
            elif axis == self.time_dimension:
                raise ValueError("The time dimension would simply be discarded after axis removal")

        return final_time_dimension

    def _get_advanced_indexed_times(self, index):
        """Get advanced indexed times based on the provided index.

        Args:
            index (np.ndarray): The index to be used for advanced indexing.

        Returns:
            tuple: A tuple containing the filtered index, timeline, and time dimension.
        """

        index = np.asarray(index)

        if index.size == 1:
            # only in case of an array containing a single value, we perform slice indexing from a numpy array
            index = index.item()
            return self._get_slice_indexed_times(index)

        else:
            # in that case, this is a boolean selection, to filter the array,
            # or an int selection, to filter and/or reorder the array
            if index.dtype == bool or index.dtype == int:

                # if it's boolean selecting on dimensions including the time_dim, we drop the timeline
                if len(index.shape) > self.time_dimension:
                    # if time dimension is the first one, we filter the time dim in the same way we do for the array
                    # and we keep the time_dimension
                    if self.time_dimension == 0:
                        return index, self.timeline[index], self.time_dimension

                        # TimelinedArray(
                        #     super().__getitem__(index),
                        #     timeline=self.timeline[index],
                        #     time_dimension=self.time_dimension,
                        # )
                    # if it has dimensions before the time dimension, then we don't know what "sub" filter to use
                    # to filter the time dimension, so we skip and return a standard array
                    else:
                        return index, None, None

                # else we return the filtered array, keeping time_dimension and timeline as is,
                # as they should be untouched
                else:
                    return index, self.timeline, self.time_dimension

                # TimelinedArray(super().__getitem__(index), timeline=self.timeline, time_dimension=self.time_dimension)
            else:
                raise ValueError(
                    "Cannot use advanced indexing with arrays " "that are not composed of either booleans or integers"
                )

    def _get_slice_indexed_times(self, index):
        """Get the indexed times based on the provided index.

        Args:
            index: Index to be used for slicing the time dimension.

        Returns:
            Tuple containing the modified index, final timeline, and the new time dimension.
        """

        # this will store the new time_dimension axis in the newly formed array
        final_time_dimension = self.time_dimension
        # this is to store the time_dimension axis to use in the index, after we searched if we have np.newaxes in it
        time_dimension_in_index = self.time_dimension

        # if we reach here, we know we are indexing either with :
        # an int for the index, a tuple of ints, a slice or a tuple of slices
        # to ease our way, we make the single int a tuple first :
        if not isinstance(index, tuple):
            index = (index,)

        # as we can add np.newaxes dynamically, we need to parse the index in a loop to defined the behaviour to adopt
        for dimension in range(len(index)):
            # as long as we are looking at indexing after the time_dimension, we don't care,
            # because standard numpy indexing will occur without changing anything about the timeline nor time_dimension
            if dimension > time_dimension_in_index:
                continue

            # np.newaxis is a placeholder for None
            if index[dimension] is None:
                # in that case, it means a dimension was added before time_dimension, so we will shift it by one.
                final_time_dimension += 1
                # we also will look at values for time dimension here to apply to timeline later
                time_dimension_in_index += 1

            # a index at time_dimension or after, is a single integer
            elif isinstance(index[dimension], (int, np.integer)):

                # if the time dimension index itself is an integer,
                # we loose time related information and return a standard numpy array
                if dimension == time_dimension_in_index:
                    return index, None, None
                    # np.array(self).__getitem__(index)

                # otherwise the dimension removed is below time_dimension,
                # and in that case, we decrease it's position in the final array
                # (not in the index, e.g. time_dimension_in_index, as this will still be used to get how to crop it)
                final_time_dimension -= 1

            # note that if index is a slice, we siply let the normal indexing occur,
            # as it doesn't add or remove dimensions

        # if a part of the index was destined to reshape the time dimension,
        # we apply this reshaping to timeline too.

        final_timeline = (
            self.timeline[index[time_dimension_in_index]] if len(index) > time_dimension_in_index else self.timeline
        )

        return index, final_timeline, final_time_dimension

    def _get_indexed_times(self, index: int | Tuple[int, ...] | slice | Tuple[slice, ...] | List | np.ndarray):
        """Get indexed times based on the provided index.

        Args:
            index (int | Tuple[int, ...] | slice | Tuple[slice] | List | np.ndarray): The index to retrieve times from.

        Returns:
            np.ndarray: The indexed times based on the provided index.
        """

        # if index is an array or a list, we do advanced indexing.
        if isinstance(index, (np.ndarray, list)):
            return self._get_advanced_indexed_times(index)
        # otherwise, if single element, (int, slice) or tuple or these, we do regular indexing.
        return self._get_slice_indexed_times(index)

    @staticmethod
    def _is_single_element(obj: np.ndarray):
        """Check if the input object is a single element.

        Args:
            obj: Input object to be checked.

        Returns:
            bool: True if the input object is a single element, False otherwise.
        """

        return obj.shape == ()

    def _finish_axis_removing_operation(
        self, result: "TimelinedArray| MaskedTimelinedArray | np.ndarray | int | float", axis: int | Tuple[int, ...]
    ) -> "TimelinedArray| MaskedTimelinedArray | np.ndarray | int | float | str":
        """Finish axis removing operation.

        Args:
            result (BaseTimeArray): The result of the operation.
            axis (int | Tuple[int, ...] | None): The axis or axes to remove.

        Returns:
            BaseTimeArray: The result after finishing the axis removing operation.
        """

        if not isinstance(result, np.ndarray) or not isinstance(result, TimeMixin):
            return result
        if self._is_single_element(result):
            return result.item()
        if self._time_dimension_in_axis(axis):
            return np.asarray(result)
        result.time_dimension = self._get_time_dimension_after_axis_removal(axis)
        return result


class BaseTimeArray(TimeMixin, np.ndarray):

    # # REDUCE and SETSTATE are used to instanciate the array from and to a pickled serialized object.
    # # We only need to store and retrieve time_dimension and timeline on top of the array's data
    def __reduce__(self):
        """Return a tuple to be used for pickling and unpickling the
        object with additional attributes 'timeline' and 'time_dimension'."""

        # Get the parent's __reduce__ tuple
        pickled_state: Tuple[Any, Any, tuple] = super().__reduce__()  # type: ignore
        # Create our own tuple to pass to __setstate__
        # type: ignore
        new_state = pickled_state[2] + (self.timeline, self.time_dimension)

        # self.logger.debug(f"Reduced to : time_dimension={self.time_dimension}. Array shape is : {new_state}")
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self: "BaseTimeArray", state):
        """Set the state of the object using the provided state tuple.

        Args:
            self (BaseTimeArray): The BaseTimeArray object.
            state: The state tuple containing information to set the object's attributes.

        Returns:
            None
        """

        self.timeline = state[-2]  # Set the info attribute
        self.time_dimension = state[-1]

        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-2])

    def __hash__(self):
        """Return the hash value of the object based on the array and timeline attributes.

        Returns:
            int: Hash value of the object.
        """

        return hash((self.__array__(), self.timeline))  # type: ignore

    def _get_array_cls(self) -> "BaseTimeArray":
        """Return the class of the array that is compatible with time operations.

        Returns:
            BaseTimeArray: The class of the array that is compatible with time operations.
        """

        valid_types = [TimelinedArray, TimelinedArray]
        for vtype in valid_types:
            if isinstance(self, vtype):
                return vtype  # type: ignore
        return np.ndarray  # type: ignore

    @property
    def array_info(self: "BaseTimeArray"):
        """Return information about the array.

        Returns:
            str: A string containing the type of the array, its shape, time dimension, and timeline shape.
        """

        return (
            f"{type(self).__name__} of shape {self.shape}, time_dimension {self.time_dimension} "
            f"and timeline shape {self.timeline.shape}"
        )

    @property
    def itime(self):
        """Return a TimeIndexer object based on the given BaseTimeArray object."""

        return TimeIndexer(self, self.start_policy, self.stop_policy)

    # backward compatibility
    isec = itime

    def align_trace(self, start: float, element_nb: int):
        """Aligns the timelined array by making it start from a timepoint in time-units (synchronizing)
        and cutting the array N elements after the start point.

        Args:
            start (float): Start point, in time-units. (usually seconds) Time index based, so can be float or integer.
            element_nb (int): Cuts the returned array at 'element_nb' amount of elements, after the starting point.
                It is item index based, not time index based, so it must necessarily be an integer.

        Returns:
            TimelinedArray: The synchronized and cut arrray.
        """

        return self.itime[start:][:element_nb]  # type: ignore

    def shift_values(
        self,
        period: int | Tuple[int, ...] | slice | Tuple[slice, ...] | List | np.ndarray = 0,
        axis=None,
        time_period=True,
    ):
        """Shifts the values of the array along the specified axis by the given period.

        Args:
            period (int): The period by which to shift the values.
            axis (int, optional): The axis along which to shift the values. Defaults to None.
            time_period (bool, optional): If True, the shift is applied based on time. Defaults to True.

        Returns:
            numpy.ndarray: The array with shifted values.
        """

        if axis is None:
            axis = self.time_dimension

        indexer = []
        for dim in range(len(self.shape)):
            if dim == axis:
                # make a new axis to account fo the axis loss of the .mean later
                indexer.append(np.newaxis)
            else:
                # select all with slice(None) equivalent to ":"
                indexer.append(slice(None))
        indexer = tuple(indexer)

        shift_area = self.itime.__getitem__(period) if time_period else self.__getitem__(period)

        # if not this : we lost a dimension because we sliced one axis to a single element, no need to no .mean
        if not len(shift_area.shape) < len(self.shape):
            shift_area = shift_area.mean(axis=axis)

        if not isinstance(shift_area, np.ndarray):
            raise NotImplementedError(
                "Returning shift_values() of the array when shift area is a scalar value (due to .mean()) "
                "is not yet implemented"
            )

        return self - np.repeat(shift_area.__getitem__(tuple(indexer)), self.shape[axis], axis=axis)

    def swapaxes(self: "BaseTimeArray", axis1: int, axis2: int):
        """Swap the two specified axes of the TimelinedArray.

        Args:
            axis1 (int): The first axis to be swapped.
            axis2 (int): The second axis to be swapped.

        Returns:
            BaseTimeArray: A new TimelinedArray with the specified axes swapped.
        """

        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        cls = self._get_array_cls()

        swapped_array: BaseTimeArray = np.swapaxes(np.asarray(self), axis1, axis2).view(cls)  # type: ignore
        swapped_array.timeline = self.timeline

        if axis1 == self.time_dimension:
            swapped_array.time_dimension = axis2
        elif axis2 == self.time_dimension:
            swapped_array.time_dimension = axis1
        else:
            swapped_array.time_dimension = self.time_dimension

        # TimelinedArray.time_dimension and TimelinedArray.timeline are set. good to go
        return swapped_array

    def transpose(self, *axes):
        """Transpose the array along the specified axes.

        Args:
            *axes: The axes to transpose the array along. If not provided, transposes the array in reverse order.

        Returns:
            BaseTimeArray: The transposed array with updated timeline and time dimension.
        """

        if not axes:
            axes = tuple(range(self.ndim))[::-1]

        cls = self._get_array_cls()

        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        transposed_array: BaseTimeArray = np.transpose(np.asarray(self), axes).view(cls)  # type: ignore
        transposed_array.timeline = self.timeline

        if self.time_dimension in axes:
            transposed_array.time_dimension = axes.index(self.time_dimension)
        else:
            transposed_array.time_dimension = self.time_dimension

        # TimelinedArray.time_dimension and TimelinedArray.timeline are set. good to go
        return transposed_array

    @property
    def T(self):
        """Transposes the object using the transpose method."""

        return self.transpose()

    def moveaxis(self: "BaseTimeArray", source: int | Tuple[int, ...], destination: int | Tuple[int, ...]):
        """Move the axis of the array to new positions.

        Args:
            source (int or Tuple[int, ...]): The source position(s) of the axis to move.
            destination (int or Tuple[int, ...]): The destination position(s) to move the axis to.

        Returns:
            BaseTimeArray: A new array with the axis moved to the specified destination.

        Note:
            This method re-instantiates a TimelinedArray with a view instead of the full
            constructor for faster performance.
        """

        if isinstance(source, int):
            source = (source,)
        if isinstance(destination, int):
            destination = (destination,)

        cls = self._get_array_cls()

        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        moved_array: BaseTimeArray = np.moveaxis(np.asarray(self), source, destination).view(cls)  # type: ignore
        moved_array.timeline = self.timeline
        moved_array.time_dimension = self.time_dimension

        if self.time_dimension in source:
            index_in_source = source.index(self.time_dimension)
            moved_array.time_dimension = destination[index_in_source]
        else:
            for src, dest in zip(source, destination):
                if src < self.time_dimension and dest >= self.time_dimension:
                    moved_array.time_dimension -= 1
                elif src > self.time_dimension and dest <= self.time_dimension:
                    moved_array.time_dimension += 1

        # TimelinedArray.time_dimension and TimelinedArray.timeline are set. good to go
        return moved_array

    def rollaxis(self: "BaseTimeArray", axis: int, start: int = 0):
        """Roll the axis of the TimelinedArray.

        Args:
            axis (int): The axis to roll.
            start (int, optional): The position where the axis is placed. Defaults to 0.

        Returns:
            BaseTimeArray: A TimelinedArray with the rolled axis.
        """

        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        cls = self._get_array_cls()

        rolled_array: BaseTimeArray = np.rollaxis(np.asarray(self), axis, start).view(cls)  # type: ignore

        # reinject timeline as is
        rolled_array.timeline = self.timeline

        # then fix the time_position according to the rolled axis

        def rollaxis_mapping(shape, axis, start=0):
            n = len(shape)
            if axis < 0:
                axis += n
            if start < 0:
                start += n
            if not (0 <= axis < n and 0 <= start <= n):
                raise ValueError("axis and start must be within valid range")
            new_order = list(range(n))

            axis_value = new_order.pop(axis)

            if start > axis:
                new_order.insert(start - 1, axis_value)
            else:
                new_order.insert(start, axis_value)
            mapping = {i: new_order.index(i) for i in range(n)}
            return mapping

        rolled_array.time_dimension = rollaxis_mapping(self.shape, axis, start)[self.time_dimension]

        return rolled_array

    def all_axes(self):
        return tuple([i for i in range(self.ndim)])

    def mean(self, axis: int | Tuple[int, ...] | None = None, dtype=None, out=None, keepdims=False):
        """Calculates the mean along the specified axis.

        Args:
            axis (int | Tuple[int, ...] | None): Axis or axes along which to perform the mean operation.
                Default is None.
            dtype: Data-type to use in the computation.
            out: Output array where the result is stored.
            keepdims (bool): If True, the reduced dimensions are retained in the output array.

        Returns:
            ndarray: Mean of the input array along the specified axis.
        """
        if axis is None:
            axis = self.all_axes()
        result = super().mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        return self._finish_axis_removing_operation(result, axis)

    # Override other reduction methods similarly if needed
    def sum(self, axis: int | Tuple[int, ...] | None = None, dtype=None, out=None, keepdims=False):
        """Calculate the sum along the specified axis.

        Args:
            axis (int | Tuple[int, ...] | None): Axis or axes along which a sum is performed.
                The default is to sum over all the dimensions of the input array.
            dtype: The type of the returned array and of the accumulator in which the elements are summed.
                If dtype is not specified, it defaults to the dtype of a, unless a has an integer dtype
                with a precision less than that of the default platform integer.
                In that case, the default platform integer is used.
            out: Alternative output array in which to place the result. It must have the same shape
                as the expected output, but the type of the output values will be cast if necessary.
            keepdims (bool): If this is set to True, the axes which are reduced are left
                in the result as dimensions with size one.
                With this option, the result will broadcast correctly against the input array.

        Returns:
            The sum of the input array along the specified axis.
        """
        if axis is None:
            axis = self.all_axes()
        result = super().sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        return self._finish_axis_removing_operation(result, axis)

    def std(
        self: "BaseTimeArray",
        axis: int | Tuple[int, ...] | None = None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
    ):
        """Calculate the standard deviation along the specified axis.

        Args:
            axis (int or Tuple[int, ...] or None): Axis or axes along which the standard deviation is computed.
                The default is to compute the standard deviation of the flattened array.
            dtype: Data-type of the result. If not provided, the data-type of the input is used.
            out: Output array with the same shape as input array, placed with the result.
            ddof (int): Delta degrees of freedom. The divisor used in calculations is N - ddof,
                where N represents the number of elements along the specified axis.
            keepdims (bool): If this is set to True, the axes which are reduced
                are left in the result as dimensions with size one.

        Returns:
            ndarray: A new array containing the standard deviation
                of elements along the specified axis after removing the axis.
        """
        if axis is None:
            axis = self.all_axes()
        result = super().std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        return self._finish_axis_removing_operation(result, axis)

    def var(
        self: "BaseTimeArray",
        axis: int | Tuple[int, ...] | None = None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
    ):
        """Calculate the variance along the specified axis.

        Args:
            self (BaseTimeArray): The input data.
            axis (int | Tuple[int, ...] | None): Axis or axes along which the variance is computed.
                The default is to compute the variance of the flattened array.
            dtype: Data-type of the result. If not provided, the data-type of the input is used.
            out: Alternative output array in which to place the result.
                It must have the same shape as the expected output but the type will be cast if necessary.
            ddof (int): Delta degrees of freedom. The divisor used in calculations is N - ddof,
                where N represents the number of elements along the specified axis.
            keepdims (bool): If this is set to True, the axes which are reduced
                are left in the result as dimensions with size one.

        Returns:
            ndarray: A new array containing the variance of the input array along the specified axis.
        """
        if axis is None:
            axis = self.all_axes()
        result = super().var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        return self._finish_axis_removing_operation(result, axis)

    def rebase_timeline(self, at=0):
        """Rebases the timeline of the array.

        Args:
            at (int): The index of the element to set as time zero. Defaults to 0.

        Returns:
            array: A modified version of the array with the timeline adjusted.
        """

        # returns a modified version of the array, with the first element of the array to time zero,
        # and shift the rest accordingly
        cls = self._get_array_cls()
        # type: ignore
        return cls(self, timeline=self.timeline - self.timeline[at])  # type: ignore

    def offset_timeline(self, offset):
        """Returns a modified version of the array with time offset.

        Args:
            offset: A fixed offset value to set time of all elements in the array relative to their current value.

        Returns:
            An array with time offset applied.

        Raises:
            None
        """

        # returns a modified version of the array, where we set time of all elements
        # in array at a fix offset relative to their current value.
        cls = self._get_array_cls()
        return cls(self, timeline=self.timeline + offset)  # type: ignore

    @property
    def pack(self):
        """Returns a TimePacker object initialized with the current instance."""

        return TimePacker(self)

    def sec_max(self):
        """Return the second maximum time from the timeline."""

        # get maximum time
        return self.timeline.max()

    def sec_min(self):
        """Get the minimum time from the timeline."""

        # get minimum time
        return self.timeline.min()

    @staticmethod
    def extract_time_from_data(data, timeline=None, time_dimension=None, uniform_space=False):
        """Extracts time-related information from the input data.

        Args:
            data: The input data from which to extract time-related information.
            timeline: The timeline associated with the data. If not provided, it will be extracted from the input data.
            time_dimension: The dimension representing time in the data. If not provided,
                it will be inferred from the input data.
            uniform_space: A boolean indicating whether the data is uniformly spaced in time.

        Returns:
            A tuple containing the processed data, timeline, and time dimension.
        """

        _unpacking = False
        # if timeline not explicitely passed as arg, we try to pick up the timeline of the input_array.
        # will rise after if input_array is not a timelined_array
        if timeline is None:
            timeline = getattr(data, "timeline", None)

        if timeline is None:
            # if arguments are an uniform list of timelined array
            # (often use to make mean and std of synchonized timelines), we pick up the first one.
            for element in data:
                timeline = getattr(element, "timeline", None)
                _unpacking = True
                break

        if timeline is None:
            raise ValueError("timeline must be supplied if the input_array is not a TimelinedArray")

        if time_dimension is None:  # same thing for the time dimension.
            time_dimension = getattr(data, "time_dimension", None)

        if time_dimension is None:
            # if arguments are an uniform list of timelined array
            # (often use to make mean and std of synchonized timelines), we pick up the first one.
            # but it also means default numpy packing will set the new dimension as dimension 0.
            # As such, the current time dimension will have to be the time dimension of the listed elements,
            # +1 (a.k.a. shifted one dimension deeper)

            for element in data:
                time_dimension = getattr(element, "time_dimension", None)
                if time_dimension is None:
                    break
                time_dimension = time_dimension + 1
                _unpacking = True
                break
            else:
                time_dimension = 0

        if time_dimension is None:
            time_dimension = 0

        if not isinstance(time_dimension, int):
            raise ValueError("time_dimension must be an integer")

        timeline = Timeline(timeline, uniform_space=uniform_space)

        if _unpacking:
            logger.debug(f"We are unpacking {type(data)} data")
            if not isinstance(data, np.ndarray) or len(data.shape) <= time_dimension:  # type: ignore
                data = np.stack(data)  # type: ignore

        return data, timeline, time_dimension


class TimePacker:
    def __init__(self, array):
        self.array = array

    def __iter__(self):
        """Returns an iterator object that contains the timeline and the array."""

        return iter((self.array.timeline, self.array.__array__()))


class TimelinedArray(BaseTimeArray):
    """
    The TimelinedArray class is a subclass of the numpy.ndarray class, which represents a multi-dimensional
    array of homogeneous data. This class adds additional functionality
    for working with arrays that have a time dimension, specifically:

    It defines a Timeline class, which is also a subclass of numpy.ndarray, and represents a timeline associated
    with the array. The Timeline class has several methods, including:
        arange_timeline: This method takes a timeline array and creates an evenly spaced timeline based
        on the start and stop time of the original timeline.
        timeline_step: This method returns the average time difference between each consecutive value in the timeline.

    TimelinedArrayIndexer class, which has several methods, including:
        time_to_index: This method converts time in seconds to index value.
        get_iindex: This method converts time in seconds to a slice object representing time.

    __new__ : This method is used to creates a new instance of the TimelinedArray class. It takes several optional
        arguments: timeline, time_dimension, arange_timeline, and timeline_is_arranged.
        It creates a TimelinedArrayIndexer object with the input array,
        and assigns the supplied timeline and dimension properties.

    It defines an indexer to access the TimelinedArray as if it was indexed by time instead of index
    It also adds an attribute time_dimension , and timeline_is_arranged to the class, which are used to keep track of
    the time dimension of the array and whether the timeline is arranged or not.
    It enables accessing the array with time instead of index, and it also tries to keep track of the time dimension
    and the timeline, so it can be used to correct indexed time.

    Example :
        ...

    """

    # backward compatibility
    TA_Timeline = Timeline

    def __new__(
        cls,
        data,
        timeline: Optional[Timeline | np.ndarray | list] = None,
        time_dimension: int | None = None,
        uniform_space=False,
    ) -> "TimelinedArray":
        """Create a new TimelinedArray object from the input data.

        Args:
            data: The input data to be stored in the TimelinedArray.
            timeline: The timeline associated with the data (default is None).
            time_dimension: The dimension representing time in the data (default is None).
            uniform_space: A boolean flag indicating if the space is uniform (default is False).

        Returns:
            TimelinedArray: A new TimelinedArray object.
        """

        data, timeline, time_dimension = BaseTimeArray.extract_time_from_data(
            data, timeline=timeline, time_dimension=time_dimension, uniform_space=uniform_space
        )

        # if np.isscalar(timeline):
        #     logger.debug(f"Scalar timeline found. Timeline is {timeline}")
        #     return np.asarray(input_array)  # type: ignore

        # instanciate the np array as a view, as per numpy documentation on how to make ndarray child classes
        obj = np.asarray(data).view(cls)

        if obj.shape[time_dimension] != len(timeline):
            raise ValueError(
                "timeline object and the shape of time_dimension of the input_array must be equal. "
                f"They are : {len(timeline)} and {obj.shape[time_dimension]}"
            )

        obj.timeline = timeline
        obj.time_dimension = time_dimension
        return obj

    def __array_finalize__(self, obj):
        """Finalize the array with additional attributes.

        Args:
            obj: Another array to finalize.

        Returns:
            None
        """

        super().__array_finalize__(obj)
        if obj is None:
            return
        self.timeline = getattr(obj, "timeline", Timeline([]))
        self.time_dimension = getattr(obj, "time_dimension", 0)

    def __array_wrap__(
        self,
        out_arr,
        context=None,
        return_scalar=False,
    ):
        """Wrap the output array after a ufunc operation.

        Args:
            out_arr: The output array to be wrapped.
            context: Additional context information (default is None).

        Returns:
            The wrapped output array.

        Example:
            If context is provided, it logs the ufunc operation name.
            If the shape of the output array is reduced, it logs the shape changes.
        """

        if context is not None:
            logger.debug(f"wrapping array after ufunc {context[0].__name__}")
        output = super().__array_wrap__(out_arr, context, return_scalar)
        if len(output.shape) < len(self.shape):
            logger.debug(f"shape reduced from : {self.shape} to : {output.shape}. outarray was : {out_arr.shape}")
        return output

    def __array_function__(self, func, types, args, kwargs):
        """Intercepts array before calling a function.

        Args:
            self: The array object.
            func: The function being called.
            types: The types of the arguments.
            args: The arguments passed to the function.
            kwargs: The keyword arguments passed to the function.

        Returns:
            The result of calling the function on the array.
        """

        logger.debug(f"intercepting array before function {func.__name__}")
        return super().__array_function__(func, types, args, kwargs)

    def __getitem__(
        self, index: int | Tuple[int, ...] | slice | Tuple[slice, ...] | List | np.ndarray
    ) -> "TimelinedArray | np.ndarray":
        """Get item from TimelinedArray based on index or slice.

        Args:
            index (int | Tuple[int, ...] | slice | Tuple[slice] | List | np.ndarray): Index or slice to retrieve item.

        Returns:
            TimelinedArray | np.ndarray: Indexed result based on the provided index.
        """

        index, final_timeline, final_time_dimension = self._get_indexed_times(index)

        if final_timeline is None or final_time_dimension is None:
            return np.asarray(self).__getitem__(index)

        indexed_result = super().__getitem__(index)

        logger.debug(
            f"Current object : {self.array_info}.\n"
            f"Newly indexed object : {type(indexed_result).__name__} of shape {indexed_result.shape}, "
            f"time_dimension {final_time_dimension} and timeline shape {final_timeline.shape}"
        )

        return TimelinedArray(indexed_result, timeline=final_timeline, time_dimension=final_time_dimension)

    # __repr__ and __str__ ARE OVERRIDEN TO AVOID HORRIBLE PERFORMANCE WHEN PRINTING
    # DUE TO CUSTOM __GETITEM__ PRE-CHECKS WITH RECURSIVE NATIVE NUMPY REPR
    def __repr__(self):
        """Return a string representation of the object with the class name and the array representation."""

        # [5:] serves to remove the 'array' part for the original array repr string
        return type(self).__name__ + np.asarray(self).__repr__()[5:]

    def __str__(self):
        """Return a string representation of the object by concatenating the class name with the string
        representation of the object as a NumPy array."""

        return type(self).__name__ + np.asarray(self).__str__()

    @staticmethod
    def align_from_iterable(iterable) -> "TimelinedArray":
        """Aligns arrays from an iterable based on their timelines.

        Args:
            iterable: An iterable containing TimelinedArray objects to align.

        Returns:
            TimelinedArray: A new TimelinedArray object containing aligned arrays.
        """

        start = max([item.timeline.min() for item in iterable])
        maxlen = min([len(item.isec[start:]) for item in iterable])

        aligned_arrays = []
        for index, item in enumerate(iterable):
            aligned_arrays.append(item.align_trace(start, maxlen))

        return TimelinedArray(aligned_arrays)


class MaskedTimelinedArray(np.ma.MaskedArray, BaseTimeArray):
    def __new__(
        cls,
        data,
        mask: NDArray[np.bool_] | np.bool_ | bool | np.ma.MaskedArray = np.ma.nomask,
        dtype=None,
        copy=False,
        fill_value=None,
        keep_mask=True,
        hard_mask=False,
        shrink=True,
        timeline: Optional[Timeline | np.ndarray | list] | None = None,
        time_dimension: Optional[int] = None,
        uniform_space=False,
        **kwargs,
    ):
        """Create a new instance of the class with the specified parameters.

        Args:
            cls: The class.
            data: The data to be used.
            mask: The mask for the data (default is np.ma.nomask).
            dtype: The data type (default is None).
            copy: Whether to copy the data (default is False).
            fill_value: The fill value for the data (default is None).
            keep_mask: Whether to keep the mask (default is True).
            hard_mask: Whether to use a hard mask (default is False).
            shrink: Whether to shrink the data (default is True).
            timeline: The timeline for the data.
            time_dimension: The time dimension for the data.
            uniform_space: Whether the space is uniform (default is False).
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of the class with the specified parameters.
        """

        _, timeline, time_dimension = BaseTimeArray.extract_time_from_data(
            data, timeline=timeline, time_dimension=time_dimension, uniform_space=uniform_space
        )

        obj = super().__new__(
            cls,
            data,
            mask=mask,
            dtype=dtype,
            copy=copy,
            fill_value=fill_value,
            keep_mask=keep_mask,
            hard_mask=hard_mask,
            shrink=shrink,
            **kwargs,
        )

        obj.timeline = timeline
        obj.time_dimension = time_dimension
        return obj

    def __array_finalize__(self, obj):
        """Finalize the array with additional attributes.

        Args:
            obj: Another array to finalize.

        Returns:
            None
        """

        super().__array_finalize__(obj)
        if obj is None:
            return
        self.timeline = getattr(obj, "timeline", Timeline([]))
        self.time_dimension = getattr(obj, "time_dimension", 0)

    def __getitem__(
        self, index: int | Tuple[int, ...] | slice | Tuple[slice] | List | np.ndarray
    ) -> "MaskedTimelinedArray | np.ma.MaskedArray":
        """Get item from the MaskedTimelinedArray based on the provided index.

        Args:
            index (int | Tuple[int, ...] | slice | Tuple[slice] | List | np.ndarray): The index or slice to retrieve.

        Returns:
            MaskedTimelinedArray | np.ma.MaskedArray: The masked array or MaskedTimelinedArray based on the index.
        """

        index, final_timeline, final_time_dimension = self._get_indexed_times(index)

        if final_timeline is None or final_time_dimension is None:
            return np.ma.MaskedArray(data=np.asarray(self), mask=self.mask, fill_value=self.fill_value).__getitem__(
                index
            )

        indexed_result = super().__getitem__(index)

        logger.debug(
            f"Current object : {self.array_info}.\n"
            f"Newly indexed object : {type(indexed_result).__name__} of shape {indexed_result.shape}, "
            f"time_dimension {final_time_dimension} and timeline shape {final_timeline.shape}"
        )

        return MaskedTimelinedArray(indexed_result, timeline=final_timeline, time_dimension=final_time_dimension)


class Seconds(float):
    def to_index(self, fs):
        """_summary_

        Args:
            fs (float or int): Sampling frequency in Hertz (samples per second)

        Returns:
            int: The samples index that this second corresponds to,
                (if sample 0 is at 0 second) in an uniformly spaced time array.
        """
        return int(self * fs)
