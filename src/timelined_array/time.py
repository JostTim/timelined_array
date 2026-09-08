from abc import abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from enum import Enum
from logging import getLogger
from typing import Any, Final, Literal, NamedTuple, Self, cast, overload
from warnings import warn

import numpy as np
import numpy.typing as npt
from numpy._typing import _UFunc_Nin2_Nout1
from numpy.typing import NDArray

logger = getLogger("timelined_array")

type PointTimeIndex = float
type SimpleTimeIndex = slice | tuple[float | None, ...]
type ComplexTimeIndex = list[float | bool] | npt.NDArray[np.floating | np.bool]
type TimeIndex = PointTimeIndex | SimpleTimeIndex | ComplexTimeIndex

all_axes: Final[None] = None
type AxisDesignation = int | tuple[int, ...] | all_axes

type Scalar = int | float | complex | str | bytes | memoryview[int]

test: np.ndarray = np.array([])


class Timeline(np.ndarray[tuple[int], np.dtype[np.float64]]):
    max_step_mult: float

    def __new__(
        cls, input_array: Iterable, uniform_space=False, max_step_mult: float = 2.0
    ):
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

        obj.max_step_mult = max_step_mult
        return obj

    def __array_finalize__(self, obj):
        """Finalize the array when subclassing a numpy array.

        Args:
            self: The subclassed array.
            obj: The original array object being subclassed.

        Returns:
            None
        """
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.max_step_mult = getattr(obj, "max_step_mult", 2.0)

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

    # def __getitem__(
    #     self: Self, index: slice | int | list | npt.NDArray
    # ) -> Self | float:
    #     result = super().__getitem__(index)
    #     if not isinstance(result, np.ndarray):
    #         return result
    #     if result.size == 1:
    #         return result.item()
    #     return result.view(self.get_class())

    # def __iter__(self) -> Iterator[float]:
    #     yield from super().__iter__()

    def get_class(self: Self) -> type[Self]:
        """Return the class of the Timeline.

        Returns:
            Timeline: the class of the timeline, or a child of that class
        """
        return self.__class__

    @classmethod
    def _uniformize(cls: type[Self], timeline: Iterable) -> Self:
        """Uniformize the given timeline data.

        Args:
            cls: The class instance.
            timeline: The timeline data to be uniformized.

        Raises:
            NotImplementedError: This function is not yet implemented.

        Returns:
            None
        """

        raise NotImplementedError(
            "Uniformization of space is not yet supported. Upcoming in future versions"
        )
        # obj = np.linspace(input_array[0], input_array[1], len(input_array)).view(cls)
        # TODO : do numpy.interp(np.arange(0, len(a), 1.5), np.arange(0, len(a)), a)
        # interp on the paretn array ? to get a fixed number of points ?
        # strategy of this is still to make up...
        # (sampling interpolation, with more/less points ? policies /arguments to make up for this)

    def uniformize(self: Self):
        """Uniformize the elements of the list using the _uniformize method."""

        self[:] = self._uniformize(self)

    @property
    def _diff(self) -> npt.NDArray[np.floating]:
        if not hasattr(self, "_cached_diff"):
            self._cached_diff = np.diff(self)
            if not (np.all(self._cached_diff >= 0) or np.all(self._cached_diff <= 0)):
                # as we don't support multi backard to forward timelines yet,
                # we make sure here that it is continuously rising or decreasing.
                # we do not do it at creating time to avoid costing compute time
                # at instanciation, and rathen have this time spend the first
                # time an actual operation involving time indexing is required
                raise ValueError(
                    "Cannot determine the step value of the timeline. "
                    "It must be strictly increasing or strictly decreasing."
                )
        return self._cached_diff

    @property
    def step(self) -> np.floating:
        """Mean time between two timeline points. Must be strictly decreasing or increasing to be calculated"""
        if not hasattr(self, "_cached_step"):
            # we get all the intervalls between consecutive time points.
            self._cached_step = np.mean(self._diff)
        return self._cached_step

    @property
    def max_step(self) -> np.floating:
        """Largest time between two timeline points, multiplied by max_step_mult (wich is )"""
        if not hasattr(self, "_cached_max_step"):
            # we get all the intervalls between consecutive time points.
            # we take the value of the largest one
            self._cached_max_step = abs(self._diff[np.argmax(np.absolute(self._diff))])
        return self._cached_max_step * self.max_step_mult

    @property
    def is_uniform(self) -> bool:
        return not any(abs(self._diff) > self.max_step)


type EdgePolicyString = Literal["inclusive", "exclusive", "inc", "exc"]


class StartEdgePolicy(Enum):
    inclusive: _UFunc_Nin2_Nout1 = np.greater_equal  # operator.ge
    exclusive: _UFunc_Nin2_Nout1 = np.greater  # operator.gt
    inc: _UFunc_Nin2_Nout1 = np.greater_equal
    exc: _UFunc_Nin2_Nout1 = np.greater


class EndEdgePolicy(Enum):
    inclusive: _UFunc_Nin2_Nout1 = np.less_equal  # operator.le
    exclusive: _UFunc_Nin2_Nout1 = np.less  # operator.lt
    inc: _UFunc_Nin2_Nout1 = np.less_equal
    exc: _UFunc_Nin2_Nout1 = np.less


class TimeIndexer[A: "BaseTimeArray"]:
    """The time indexer indexes by default from >= to the time start, and strictly < to time stop"""

    def __init__(
        self,
        array: A,
        start: EdgePolicyString = "inclusive",
        stop: EdgePolicyString = "exclusive",
    ):
        self.array = array
        self.set_edge_policy(start, stop)

    def set_edge_policy(
        self: Self,
        start: EdgePolicyString = "inclusive",
        stop: EdgePolicyString = "exclusive",
    ) -> Self:
        self.start_operation = StartEdgePolicy[start].value
        self.stop_operation = EndEdgePolicy[stop].value
        return self

    @overload
    def time_to_index(self, time: PointTimeIndex) -> int: ...

    @overload
    def time_to_index(self, time: SimpleTimeIndex) -> slice[int, int, int]: ...

    @overload
    def time_to_index(self, time: ComplexTimeIndex) -> npt.NDArray[np.integer]: ...

    def time_to_index(
        self, time: TimeIndex
    ) -> int | slice[int, int, int] | npt.NDArray[np.integer]:
        """Converts time to index with methods based on the different input types.

        Args:
            time (float | int | slice | Tuple[int | float] | List[float | int | slice | Tuple[int | float]]):
                The time value or range to be converted to index.

        Returns:
            int: The index corresponding to the input time value or range.

        Raises:
            ValueError: If the input time type is not supported.
        """
        # argument index may be a slice or a scalar. Units of index should be in second. Returns a slice as index
        # this is sort of a wrapper for get_time_slice that does the heavy lifting.
        # this function just makes sure to pass arguments to it corectly depending
        # on if the time index is a single value or a slice.

        if isinstance(time, slice):
            # if we get a slice of times, we return a slice of indices
            return self.get_time_slice(time.start, time.stop, time.step)
        elif isinstance(time, tuple):
            # if we get a tuple of times (up to size = 3) we interpret it as a slice
            # meaning : (start, stop, step)
            if len(time) > 3:
                raise ValueError(
                    "In case we index with a tuple, it's length must be at "
                    f"maximum 3 elements (start, stop, step). Found {len(time)} elements"
                )
            return self.get_time_slice(*time)
        elif isinstance(time, (list, np.ndarray)):
            # if we get a list or an array, we return a list where
            # each element is an index to be taken, corresponding to that point in the timeline.
            # for now we use the start_edge polici to take these indices
            # (so index corresponding to each time point either > or >= to time point, depending on
            # inclusive / exclusive policy).
            return np.array(self.get_start_index(t) for t in time)

        # finally, the last possible case is that the time is a scalar (int / float etc)
        # if not, we let the program crash and report the normal error to the user.
        # if it is, we return the start of the slice index, corresponding to that point in time
        return self.get_time_slice(time_start=time).start  # in that case we return int

    @property
    def max_step(self) -> np.floating:
        return abs(self.array.timeline.max_step)

    seconds_to_index = time_to_index

    def insert_time_index_into_full_index[T: int | slice[int, int, int] | np.ndarray](
        self, time_index: T
    ) -> tuple[slice[None] | T, ...]:
        """Inserts a time index into the full index.
        Becauer Time indexer is supposed to work only on the time
        dimension, the indexing of the rest of the dimensions is made only of
        slice(None) wich corresponds to [:] in usual indexing syntax.

        Args:
            time_index: The index to be inserted into the full index.

        Returns:
            tuple: The full index with the time index inserted.
        """
        # first we get a list of [slice(None), ...] as much times as there is dimensions
        full_index: list[slice[None] | T] = [slice(None)] * len(self.array.shape)

        # then we put the integer value at the position of time index at
        # the right position (time_dimension) in the tuple of all sliced dimensions
        full_index[self.array.time_dimension] = time_index

        return tuple(full_index)

    @overload
    def __getitem__(self, index: PointTimeIndex) -> float: ...

    @overload
    def __getitem__(self, index: SimpleTimeIndex | ComplexTimeIndex) -> A: ...

    def __getitem__(self, index: TimeIndex) -> A | float:
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

        index_time = self.time_to_index(index)
        full_iindex = self.insert_time_index_into_full_index(index_time)

        logger.debug(
            f"About to index over time with iindex_time {index_time} and full_iindex {full_iindex}"
        )
        return cast(A | float, self.array[full_iindex])

    def check_step_size_below_maximum(self, index: int, time_index: float) -> None:

        if abs(self.array.timeline[index] - time_index) > self.max_step:
            raise IndexError(
                f"The start time value {time_index} you searched for is not in the timeline of this array "
                f"(timeline starts at {self.array.timeline[0]} and ends at {self.array.timeline[-1]}, "
                f"allowed jitter = {self.max_step} : "
                " +/- 2 times the max step between two timeline points"
            )

    def get_start_index(self, time_start: float | None) -> int:
        if time_start is None:
            # if start is none, we want to start at the first element of the array no matter what,
            # so we do not check wether the time start is actually a lot before the first time of the timeline
            return 0

        # if time_start is > or >= (depending in start_operation than self.array.timeline[0],
        # then we use that to determine the actual index closest to the supplied time)
        if self.start_operation(time_start, self.array.timeline[0]):
            start = int(
                np.argmax(self.start_operation(self.array.timeline, time_start))
            )
        else:
            # else, it meant the start time is not even in the array, so we use the start,
            # and we check if the supplied time was before the time at index 0, more than the max step
            start = 0

        self.check_step_size_below_maximum(start, time_start)
        return start

    def get_stop_index(self, time_stop: float | None) -> int:
        if time_stop is None:
            # if stop is none, we want to stop at the very last element of the array no matter what,
            # so we do not check wether the time stop is actually a lot after the last time of the timeline
            return len(self.array.timeline)

        # using the end stop edge policy to determine the index corresponding to the given end_time
        if self.stop_operation(time_stop, self.array.timeline[-1]):
            stop = int(np.argmin(self.stop_operation(self.array.timeline, time_stop)))
        else:
            # the last element of the array, we will check next with check_step_size_below_maximum
            # to see if it is too far from the actual time asked (time_stop) in wich case we raise an error.
            # to use it to take all the times regardless of wether it's far or not from the asked time, one should
            # use : (wich is equivalent to None in the slice.stop object)
            stop = len(self.array.timeline)

        self.check_step_size_below_maximum(stop, time_stop)
        return stop

    def get_step_index(self, time_step: float | None) -> int:
        if time_step is None:
            return 1

        step = int(np.round(time_step / self.array.timeline.step))
        return max(step, 1)

    def get_time_slice(
        self,
        time_start: float | None = None,
        time_stop: float | None = None,
        time_step: float | None = None,
    ) -> slice[int, int, int]:
        """Get the index range based on the given start, stop, and step values in seconds.

        Args:
            sec_start (float): The start time, in your time units. If None, start index will be  will be set to the first point of the timeline (0).
            sec_stop (float): The stop time, in your time units. If None, the stop index will be set to the final point of the timeline.
            sec_step (float): The step size, in your time units. If None, the index step size will be 1.

        Returns:
            slice: A slice object representing the index range based on the given start, stop, and step values.
        """
        # converts a time index (follows a slice syntax, but in time units) to integer units

        return slice(
            self.get_start_index(time_start),
            self.get_stop_index(time_stop),
            self.get_step_index(time_step),
        )

    def __call__(
        self: Self,
        start: EdgePolicyString = "inclusive",
        stop: EdgePolicyString = "exclusive",
    ) -> Self:
        return self.set_edge_policy(start, stop)


@dataclass(frozen=True)
class TimeAttributes:
    timeline: Timeline
    time_dimension: int
    collapsed: bool = False

    @classmethod
    def from_timeline_array(cls: type[Self], array: "BaseTimeArray") -> Self:
        return cls(array.timeline, array.time_dimension)

    def apply_to_array[T: BaseTimeArray](
        self, cls: type[T], array: npt.NDArray
    ) -> T | npt.NDArray:

        if self.collapsed:
            array = array.__array__()
            if array.size == 1:
                return array.item()
            return array

        # we reinstanciate the newly created array with a view, as this is faster than a brand __new__
        output_array = array.view(cls)
        output_array.timeline = self.timeline
        output_array.time_dimension = self.time_dimension
        return output_array


class IndexationManager:
    def __init__(self, array: "BaseTimeArray") -> None:
        self.array = array

    def apply_index_to_time_attrs(
        self,
        time_axis_index: int | npt.NDArray | list | slice | None,
        time_attrs: TimeAttributes,
    ) -> TimeAttributes:
        """This function applies the array indexing to the timeline. This function **assumes** that the
        given array applies to the timeline. (the check should be done outside of this function)
        Wether it applies boolean based array indexing or int based array indexing is dealt with
        by the parent numpy indexing resolvers)."""
        if time_axis_index is np.newaxis:
            raise ValueError(
                "It should be impossible to find np.newaxis at this stage."
            )
        if isinstance(time_axis_index, int):
            # in that case, the time dimension collapses, we return None
            return replace(time_attrs, collapsed=True)

        return replace(time_attrs, timeline=time_attrs.timeline[time_axis_index])

    def expand_new_axes(
        self, index: tuple[None | Any, ...], time_attrs: TimeAttributes
    ) -> TimeAttributes:
        """In case we have None (equivalent to np.newaxes) before or at equal axis
        positions than the current time_dimension, we "raise" it's position by 1"""
        for axis, axis_index in enumerate(index):
            if axis_index is np.newaxis and axis <= time_attrs.time_dimension:
                time_attrs = replace(
                    time_attrs, time_dimension=time_attrs.time_dimension + 1
                )
        return time_attrs

    def reduce_collapsed_axes(
        self, index: tuple[int | Any, ...], time_attrs: TimeAttributes
    ) -> TimeAttributes:
        """After running other operations, if we notice that we have "singular" points in the axis, we
        "lower" the current time_dimension by 1"""
        if time_attrs.collapsed:
            return time_attrs
        for axis, axis_index in enumerate(index):
            if np.isscalar(axis_index) and axis < time_attrs.time_dimension:
                time_attrs = replace(
                    time_attrs, time_dimension=time_attrs.time_dimension - 1
                )
        return time_attrs

    def tuple_indexing(
        self,
        index: tuple[int | None | slice | npt.NDArray | list, ...],
        time_attrs: TimeAttributes,
    ) -> TimeAttributes:
        # first, if there is newwaxes, we shift the time_dimension upwards
        l_time_attrs = self.expand_new_axes(index, time_attrs)

        if len(index) > l_time_attrs.time_dimension:
            l_time_attrs = self.apply_index_to_time_attrs(
                index[l_time_attrs.time_dimension], l_time_attrs
            )

        # last, if there is axes collapsing, we shift the time_dimension downwards
        return self.reduce_collapsed_axes(index, l_time_attrs)

    def array_indexing(
        self, index: npt.NDArray | list, time_attrs: TimeAttributes
    ) -> TimeAttributes:
        index = np.asarray(index)

        if index.ndim > 1:
            warn(
                "We do not yet support timeline management "
                "with complex numpy arrays indices, of more thqn 1 dimension."
                "Returning a normal numpy array instead of a time array",
                FutureWarning,
            )
            return replace(time_attrs, collapsed=True)

        if index.shape[0] != time_attrs.timeline.shape[0]:
            # if the array indexing does not "concern" the time dimension, we don't change the time attributes
            return time_attrs
        return self.apply_index_to_time_attrs(index, time_attrs)

    # was _get_indexed_times(
    def manage(
        self,
        index: None
        | int
        | slice
        | tuple[int | None | slice | npt.NDArray | list, ...]
        | list
        | npt.NDArray,
    ) -> TimeAttributes:
        time_attrs = TimeAttributes.from_timeline_array(self.array)

        """Get indexed times based on the provided index.

        Args:
            index (int | Tuple[int, ...] | slice | Tuple[slice] | List | np.ndarray): The index to retrieve times from.

        Returns:
            np.ndarray: The indexed times based on the provided index.
        """

        if isinstance(index, (np.ndarray, list)):
            return self.array_indexing(index, time_attrs)
        elif index is None or isinstance(index, (int, slice)):
            return self.tuple_indexing((index,), time_attrs)
        elif isinstance(index, tuple):
            return self.tuple_indexing(index, time_attrs)
        raise TypeError(f"Cannot index the array with type {type(index)}")


class CollapseOperationManager:
    # the role of this class is to manage the time dimension
    # of the timeline, when an array undergoes dimensionally reducing operations (such as mean, sum, etc)

    def __init__(self, array: "BaseTimeArray") -> None:
        self.array = array

    def time_dimension_in_axis(
        self, collapsed_axes: AxisDesignation, time_attrs: TimeAttributes
    ) -> bool:
        """Check if the time dimension is present in the specified axis.

        Args:
            axis (int | Tuple[int, ...] | None): The axis to check for the time dimension.

        Returns:
            bool: True if the time dimension is present in the axis, False otherwise.
        """

        return bool(
            collapsed_axes is all_axes
            or collapsed_axes == time_attrs.time_dimension
            or (
                isinstance(collapsed_axes, tuple)
                and time_attrs.time_dimension in collapsed_axes
            )
        )

    def reduce_collapsed_axes(
        self, collapsed_axes: int | tuple[int, ...], time_attrs: TimeAttributes
    ) -> TimeAttributes:
        """Return the time dimension after removing specified axis.

        Args:
            axis_removed (int or tuple): The axis or axes to be removed.

        Returns:
            int: The time dimension after removing the specified axis or axes.

        Raises:
            ValueError: If the time dimension would be discarded after axis removal.
        """

        if not isinstance(collapsed_axes, tuple):
            collapsed_axes = (collapsed_axes,)
        axis_removed = sorted(collapsed_axes)

        axis_reduced_before_time_axis = tuple(
            axis for axis in axis_removed if axis < time_attrs.time_dimension
        )
        return replace(
            time_attrs,
            time_dimension=time_attrs.time_dimension
            - len(axis_reduced_before_time_axis),
        )

    def manage(
        self,
        result: npt.NDArray,
        collapsed_axes: AxisDesignation,
    ) -> TimeAttributes:
        """Finish axis removing operation.

        Args:
            result (BaseTimeArray): The result of the operation.
            axis (int | Tuple[int, ...] | None): The axis or axes to remove.

        Returns:
            BaseTimeArray: The result after finishing the axis removing operation.
        """

        time_attrs = TimeAttributes.from_timeline_array(self.array)

        if (
            np.isscalar(result)
            or (isinstance(result, np.ndarray) and result.size == 1)
            or not isinstance(result, np.ndarray)
            or self.time_dimension_in_axis(collapsed_axes, time_attrs)
        ):
            return replace(time_attrs, collapsed=True)
        return self.reduce_collapsed_axes(collapsed_axes, time_attrs)  # type : ignore


class BaseTimeArray(np.ndarray):  # All time arrays are numpy arrays
    # # REDUCE and SETSTATE are used to instanciate the array from and to a pickled serialized object.
    # # We only need to store and retrieve time_dimension and timeline on top of the array's data
    timeline: Timeline
    time_dimension: int

    @property
    def itime(self: Self) -> TimeIndexer[Self]:
        """Return a TimeIndexer object based on the given BaseTimeArray object."""

        return TimeIndexer(self)

    @property
    def pack(self: Self) -> "TimePacker":
        """Returns a TimePacker object initialized with the current instance."""

        return TimePacker(self.timeline, self.__array__())

    def align_trace(self: Self, start: float, element_nb: int) -> Self:
        """Aligns the timelined array by making it start from a timepoint in time-units (synchronizing)
        and cutting the array N elements after the start point.

        Args:
            start (float): Start point, in time-units. (usually seconds) Time index based, so can be float or integer.
            element_nb (int): Cuts the returned array at 'element_nb' amount of elements, after the starting point.
                It is item index based, not time index based, so it must necessarily be an integer.

        Returns:
            TimelinedArray: The synchronized and cut arrray.
        """
        # this slice is to select the number of elements, on the time_dimension
        slices = tuple(
            slice(None) if i != self.time_dimension else slice(None, element_nb)
            for i in range(self.ndim)
        )

        return cast(Self, self.itime[start:][slices])

    @staticmethod
    def extract_time_from_data(
        data, timeline=None, time_dimension=None, uniform_space=False
    ) -> tuple[NDArray, Timeline, int]:
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
            raise ValueError(
                "timeline must be supplied if the input_array is not a TimelinedArray"
            )

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
            raise TypeError("time_dimension must be an integer")

        timeline = Timeline(timeline, uniform_space=uniform_space)

        if _unpacking:
            logger.debug(f"We are unpacking {type(data)} data")
            if not isinstance(data, np.ndarray) or len(data.shape) <= time_dimension:
                data = np.stack(data)

        return data, timeline, time_dimension

    @classmethod
    def align_from_iterable(
        cls: type[Self], iterable: "Iterable[BaseTimeArray]"
    ) -> Self:
        """Aligns arrays from an iterable based on their timelines.

        Args:
            iterable: An iterable containing TimelinedArray objects to align.

        Returns:
            TimelinedArray: A new TimelinedArray object containing aligned arrays.
        """

        start = max(item.timeline.min() for item in iterable)
        maxlen = min([len(item.itime[start:]) for item in iterable])

        aligned_arrays = [item.align_trace(start, maxlen) for item in iterable]
        return cls(aligned_arrays)

    def max_time(self) -> float:  # was sec_max
        """Return the second maximum time from the timeline."""

        # get maximum time
        return self.timeline.max()

    def min_time(self) -> float:  # was sec_min
        """Get the minimum time from the timeline."""

        # get minimum time
        return self.timeline.min()

    def rebase_timeline(self: Self, at: float = 0.0) -> Self:
        """Rebases the timeline of the array.

        Args:
            at (int): The index of the element to set as time zero. Defaults to 0.

        Returns:
            array: A modified version of the array with the timeline adjusted.
        """

        # returns a modified version of the array, with the first element
        #  of the array to time zero, and shift the rest accordingly
        new_array = self.view(self.get_class())
        new_array.timeline = Timeline((self.timeline - self.timeline.min()) + at)
        return new_array

    def offset_timeline(self: Self, offset: float) -> Self:
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
        new_array = self.view(self.get_class())
        new_array.timeline = Timeline(self.timeline + offset)
        return new_array

    def get_class(self: Self) -> type[Self]:
        """Return the class of the array that is compatible with time operations.

        Returns:
            BaseTimeArray: The class of the array that is compatible with time operations.
        """
        return self.__class__

    def get_class_name(self) -> str:
        return self.get_class().__name__

    @classmethod
    @abstractmethod
    def __as_time_unaware__(
        cls,
        array: npt.NDArray,
    ) -> npt.NDArray: ...

    def __array_finalize__(self, obj: NDArray | None) -> None:
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

    def __reduce__(
        self,
    ) -> tuple[Any, Any, tuple[Any, Any, Any, Any, Any, Timeline, int]]:
        """Return a tuple to be used for pickling and unpickling the
        object with additional attributes 'timeline' and 'time_dimension'."""
        # Get the parent's __reduce__ tuple
        pickled_state: tuple[Any, Any, tuple] = super().__reduce__()  # type: ignore
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.timeline, self.time_dimension)

        # self.logger.debug(f"Reduced to : time_dimension={self.time_dimension}. Array shape is : {new_state}")
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(
        self: Self, state: tuple[Any, Any, Any, Any, Any, Timeline, int]
    ) -> None:
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

    def __hash__(self) -> int:
        """Return the hash value of the object based on the array and timeline attributes.

        Returns:
            int: Hash value of the object.
        """

        return hash((self.__array__(), self.timeline))

    # __repr__ and __str__ ARE OVERRIDEN TO AVOID HORRIBLE PERFORMANCE WHEN PRINTING
    # DUE TO CUSTOM __GETITEM__ : PRE-CHECKS WITH RECURSIVE NATIVE NUMPY REPR
    def __repr__(self) -> str:
        """Return a string representation of the object with the class name and the array representation."""

        # [5:] serves to remove the 'array' part for the original array repr string
        return self.get_class_name() + self.__as_time_unaware__(self).__repr__()[5:]

    def __str__(self) -> str:
        """Return a string representation of the object by concatenating the class name with the string
        representation of the object as a NumPy array."""

        return self.get_class_name() + self.__as_time_unaware__(self).__str__()

    def __getitem__(
        self: Self,
        index: None
        | int
        | slice
        | tuple[int | None | slice | npt.NDArray | list, ...]
        | list
        | npt.NDArray,
    ) -> Self | npt.NDArray:
        """Get item from TimelinedArray based on index or slice.

        Args:
            index (int | Tuple[int, ...] | slice | Tuple[slice] | List | np.ndarray): Index or slice to retrieve item.

        Returns:
            TimelinedArray | np.ndarray: Indexed result based on the provided index.
        """

        time_attrs = IndexationManager(self).manage(index)

        if time_attrs.collapsed:
            result = self.__as_time_unaware__(self).__getitem__(index)
            # if isinstance(result, np.ndarray) and result.size == 1:
            #     return result.item()
            return result

        result = super().__getitem__(index)
        return time_attrs.apply_to_array(self.get_class(), result)

    def __iter__(self: Self) -> Iterator[Self | npt.NDArray | Scalar]:
        """Iterate over the first axis using numpy's C-level iterator,
        instead of falling back to the (much slower) overloaded __getitem__.
        The yielded sub-arrays keep a coherent timeline : as the first axis
        is consumed, the time dimension moves one position up, and is dropped
        (yielding plain arrays) when the time dimension is the iterated axis."""

        for item in super().__iter__():
            if self.time_dimension == 0:
                if self.ndim > 1:
                    yield np.asarray(item)
                else:
                    yield item
            else:
                item.time_dimension = self.time_dimension - 1
                yield item

    def swapaxes(self: Self, axis1: int, axis2: int) -> Self:
        """Swap the two specified axes of the TimelinedArray.

        Args:
            axis1 (int): The first axis to be swapped.
            axis2 (int): The second axis to be swapped.

        Returns:
            BaseTimeArray: A new TimelinedArray with the specified axes swapped.
        """

        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        cls = self.get_class()

        swapped_array = np.swapaxes(np.asarray(self), axis1, axis2).view(cls)
        swapped_array.timeline = self.timeline

        if axis1 == self.time_dimension:
            swapped_array.time_dimension = axis2
        elif axis2 == self.time_dimension:
            swapped_array.time_dimension = axis1
        else:
            swapped_array.time_dimension = self.time_dimension

        # TimelinedArray.time_dimension and TimelinedArray.timeline are set. good to go
        return swapped_array

    def transpose(self: Self, *axes) -> Self:
        """Transpose the array along the specified axes.

        Args:
            *axes: The axes to transpose the array along. If not provided, transposes the array in reverse order.

        Returns:
            BaseTimeArray: The transposed array with updated timeline and time dimension.
        """

        if not axes:
            axes = tuple(range(self.ndim))[::-1]

        cls = self.get_class()

        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        transposed_array = np.transpose(np.asarray(self), axes).view(cls)
        transposed_array.timeline = self.timeline

        if self.time_dimension in axes:
            transposed_array.time_dimension = axes.index(self.time_dimension)
        else:
            transposed_array.time_dimension = self.time_dimension

        # TimelinedArray.time_dimension and TimelinedArray.timeline are set. good to go
        return transposed_array

    @property
    def T(self: Self) -> Self:
        """Transposes the object using the transpose method."""

        return self.transpose()

    def moveaxis(
        self: Self,
        source: int | tuple[int, ...],
        destination: int | tuple[int, ...],
    ) -> Self:
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

        cls = self.get_class()

        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        moved_array = np.moveaxis(np.asarray(self), source, destination).view(cls)
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

    def rollaxis(self: Self, axis: int, start: int = 0) -> Self:
        """Roll the axis of the TimelinedArray.

        Args:
            axis (int): The axis to roll.
            start (int, optional): The position where the axis is placed. Defaults to 0.

        Returns:
            BaseTimeArray: A TimelinedArray with the rolled axis.
        """

        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        cls = self.get_class()

        rolled_array = np.rollaxis(np.asarray(self), axis, start).view(cls)

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

        rolled_array.time_dimension = rollaxis_mapping(self.shape, axis, start)[
            self.time_dimension
        ]

        return rolled_array

    def mean(
        self: Self,
        axis: int | tuple[int, ...] | None = None,
        dtype=None,
        out=None,
        keepdims=False,
    ) -> Self | np.ndarray | float:
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

        result = super().mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        time_attrs = CollapseOperationManager(self).manage(result, axis)
        return time_attrs.apply_to_array(self.get_class(), result)

    # Override other reduction methods similarly if needed
    def sum(
        self: Self,
        axis: int | tuple[int, ...] | None = None,
        dtype=None,
        out=None,
        keepdims=False,
    ) -> Self | np.ndarray | float:
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

        result = super().sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        time_attrs = CollapseOperationManager(self).manage(result, axis)
        return time_attrs.apply_to_array(self.get_class(), result)

    def std(
        self: Self,
        axis: int | tuple[int, ...] | None = None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
    ) -> Self | np.ndarray | float:
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

        result = super().std(
            axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims
        )
        time_attrs = CollapseOperationManager(self).manage(result, axis)
        return time_attrs.apply_to_array(self.get_class(), result)

    def var(
        self: Self,
        axis: int | tuple[int, ...] | None = None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
    ) -> Self | np.ndarray | float:
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
        result = super().var(
            axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims
        )
        time_attrs = CollapseOperationManager(self).manage(result, axis)
        return time_attrs.apply_to_array(self.get_class(), result)


class TimePacker(NamedTuple):
    time: Timeline
    data: npt.NDArray


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
        get_time_slice: This method converts time in seconds to a slice object representing time.

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

    def __new__(
        cls: type[Self],
        data: npt.ArrayLike,
        timeline: Timeline | np.ndarray | list | None = None,
        time_dimension: int | None = None,
        uniform_space=False,
    ) -> Self:
        """Create a new TimelinedArray object from the input data.

        Args:
            data: The input data to be stored in the TimelinedArray.
            timeline: The timeline associated with the data (default is None).
            time_dimension: The dimension representing time in the data (default is None).
            uniform_space: A boolean flag indicating if the space is uniform (default is False).

        Returns:
            TimelinedArray: A new TimelinedArray object.
        """

        data, timeline, time_dimension = cls.extract_time_from_data(
            data,
            timeline=timeline,
            time_dimension=time_dimension,
            uniform_space=uniform_space,
        )

        # instanciate the standard np array as a view, as per numpy
        # documentation on how to make ndarray child classes
        obj = np.asarray(data).view(cls)

        if obj.shape[time_dimension] != len(timeline):
            raise ValueError(
                "timeline object and the shape of time_dimension of the input_array must be equal. "
                f"They are : {len(timeline)} and {obj.shape[time_dimension]}"
            )

        obj.timeline = timeline
        obj.time_dimension = time_dimension
        return obj

    @classmethod
    def __as_time_unaware__(self, array: npt.NDArray) -> np.ndarray:
        return np.asarray(array)


class MaskedTimelinedArray(BaseTimeArray, np.ma.MaskedArray):
    def __new__(
        cls: type[Self],
        data,
        mask: NDArray[np.bool_] | np.bool_ | bool | np.ma.MaskedArray = np.ma.nomask,
        dtype=None,
        copy=False,
        fill_value=None,
        keep_mask=True,
        hard_mask=False,
        shrink=True,
        timeline: Timeline | np.ndarray | list | None = None,
        time_dimension: int | None = None,
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

        _, timeline, time_dimension = cls.extract_time_from_data(
            data,
            timeline=timeline,
            time_dimension=time_dimension,
            uniform_space=uniform_space,
        )

        obj = (
            super()
            .__new__(
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
            .view(cls)
        )

        obj.timeline = timeline
        obj.time_dimension = time_dimension
        return obj

    @classmethod
    def __as_time_unaware__(
        cls,
        array: npt.NDArray,
    ) -> np.ma.MaskedArray:
        return np.ma.asarray(array)
