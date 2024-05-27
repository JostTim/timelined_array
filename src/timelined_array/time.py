# -*- coding: utf-8 -*-

import numpy as np
from logging import getLogger
from typing import Tuple, List, Protocol, Type


logger = getLogger("timelined_array")


class TimeCompatibleProtocol(Protocol):

    time_dimension: int
    timeline: "Timeline"

    def __getitem__(self, index) -> np.ndarray: ...

    def __array__(self) -> np.ndarray: ...

    @property
    def shape(self) -> Tuple[int]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def itime(self) -> "TimeIndexer": ...

    def _get_array_cls(self) -> "Type": ...

    def transpose(self): ...

    def _finish_axis_removing_operation(self, result, axis): ...


class Timeline(np.ndarray):
    def __new__(cls, input_array, uniform_space=False):

        if uniform_space:
            # if we want a uniformly spaced timeline from start to stop of the current timeline.
            obj = Timeline._uniformize(input_array)
        else:
            if isinstance(input_array, Timeline):
                return input_array
            obj = np.asarray(input_array).view(cls)

        return obj

    def __array_finalize__(self, obj):
        pass

    def __setstate__(self, state):
        # try:
        #     super().__setstate__(state[0:-2])  # old deserializer
        # except TypeError:
        super().__setstate__(state)  # new one

    def __contains__(self, time_value):
        return self.min() <= time_value <= self.max()

    def max(self):
        return super().max().item()

    def min(self):
        return super().min().item()

    @classmethod
    def _uniformize(cls, timeline):
        raise NotImplementedError("Upcoming function")
        # obj = np.linspace(input_array[0], input_array[1], len(input_array)).view(cls)
        # TODO : do numpy.interp(np.arange(0, len(a), 1.5), np.arange(0, len(a)), a)
        # interp to get a fixed number of points ?

    def uniformize(self):
        self[:] = self._uniformize(self)


class TimeIndexer:
    """The time indexer indexes by default from >= to the time start, and strictly < to time stop"""

    def __init__(self, array: TimeCompatibleProtocol):
        self.array = array

    def seconds_to_index(self, index):
        # argument index may be a slice or a scalar. Units of index should be in second. Returns a slice as index
        # this is sort of a wrapper for get_iindex that does the heavy lifting.
        # this function just makes sure to pass arguments to it corectly depending
        # on if the time index is a single value or a slice.
        if isinstance(index, slice):
            return self.get_iindex(index.start, index.stop, index.step)
        else:
            return self.get_iindex(sec_start=index).start

    def _insert_time_index(self, time_index):
        # put the integer value at the position of time index at the right position
        # (time_dimension) in the tuple of all sliced dimensions
        full_index = [slice(None)] * len(self.array.shape)
        full_index[self.array.time_dimension] = time_index

        return tuple(full_index)

    def __getitem__(self, index) -> "TimelinedArray | MaskedTimelinedArray | np.ndarray":
        if hasattr(index, "__iter__"):
            # if not isinstance(index,(int,float,slice,np.integer,np.floating)):
            raise ValueError(
                "Isec allow only indexing on time dimension. Index must be either int, float or slice, not iterable"
            )

        iindex_time = self.seconds_to_index(index)
        full_iindex = self._insert_time_index(iindex_time)
        # print("new full index : ",iindex_time)
        logger.debug(f"About to index over time with iindex_time {iindex_time} and full_iindex {full_iindex}")
        return self.array[full_iindex]

    def get_iindex(self, sec_start=None, sec_stop=None, sec_step=None):  # every value here is in seconds
        # converts a time index (follows a slice syntax, but in time units) to integer units
        timeline_max_step = np.absolute(np.diff(self.array.timeline)).max() * 2

        if sec_start is None:
            start = 0
        else:
            if sec_start >= self.array.timeline[0]:
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
            if sec_stop < self.array.timeline[-1]:
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


class TimeMixin:

    time_dimension: int
    timeline: Timeline

    def _time_dimension_in_axis(self, axis: int | Tuple[int] | None) -> bool:
        if (
            axis is None
            or axis == self.time_dimension
            or (isinstance(axis, (list, tuple)) and self.time_dimension in axis)
        ):
            return True
        return False

    def _get_time_dimension_after_axis_removal(self, axis_removed) -> int:
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

    def _get_indexed_times(self, index: int | Tuple[int] | slice | Tuple[slice] | List | np.ndarray):

        # if index is an array or a list, we do advanced indexing.
        if isinstance(index, (np.ndarray, list)):
            return self._get_advanced_indexed_times(index)
        # otherwise, if single element, (int, slice) or tuple or these, we do regular indexing.
        return self._get_slice_indexed_times(index)

    @staticmethod
    def _is_single_element(obj):
        return obj.shape == ()

    def _finish_axis_removing_operation(self, result: TimeCompatibleProtocol, axis: int | Tuple[int] | None):
        if not isinstance(result, np.ndarray):
            return result
        if not self._is_single_element(result):
            return result.item()
        if self._time_dimension_in_axis(axis):
            return np.asarray(result)
        result.time_dimension = self._get_time_dimension_after_axis_removal(axis)
        return result

    # # REDUCE and SETSTATE are used to instanciate the array from and to a pickled serialized object.
    # # We only need to store and retrieve time_dimension and timeline on top of the array's data
    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.timeline, self.time_dimension)  # type: ignore

        # self.logger.debug(f"Reduced to : time_dimension={self.time_dimension}. Array shape is : {new_state}")
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self: TimeCompatibleProtocol, state):
        self.timeline = state[-2]  # Set the info attribute
        self.time_dimension = state[-1]

        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-2])

    def __hash__(self):
        return hash((self.__array__(), self.timeline))  # type: ignore

    def _get_array_cls(self) -> TimeCompatibleProtocol:
        valid_types = [TimelinedArray, TimelinedArray]
        for vtype in valid_types:
            if isinstance(self, vtype):
                return vtype  # type: ignore
        return np.ndarray  # type: ignore

    @property
    def array_info(self: TimeCompatibleProtocol):
        return (
            f"{type(self).__name__} of shape {self.shape}, time_dimension {self.time_dimension} "
            f"and timeline shape {self.timeline.shape}"
        )

    @property
    def itime(self: TimeCompatibleProtocol):
        return TimeIndexer(self)

    @property
    def isec(self: TimeCompatibleProtocol):
        return self.itime

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

    def swapaxes(self: TimeCompatibleProtocol, axis1: int, axis2: int):
        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        cls = self._get_array_cls()

        swapped_array: TimeCompatibleProtocol = np.swapaxes(np.asarray(self), axis1, axis2).view(cls)  # type: ignore
        swapped_array.timeline = self.timeline

        if axis1 == self.time_dimension:
            swapped_array.time_dimension = axis2
        elif axis2 == self.time_dimension:
            swapped_array.time_dimension = axis1
        else:
            swapped_array.time_dimension = self.time_dimension

        # TimelinedArray.time_dimension and TimelinedArray.timeline are set. good to go
        return swapped_array

    def transpose(self: TimeCompatibleProtocol, *axes):
        if not axes:
            axes = tuple(range(self.ndim))[::-1]

        cls = self._get_array_cls()

        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        transposed_array: TimeCompatibleProtocol = np.transpose(np.asarray(self), axes).view(cls)  # type: ignore
        transposed_array.timeline = self.timeline

        if self.time_dimension in axes:
            transposed_array.time_dimension = axes.index(self.time_dimension)
        else:
            transposed_array.time_dimension = self.time_dimension

        # TimelinedArray.time_dimension and TimelinedArray.timeline are set. good to go
        return transposed_array

    @property
    def T(self: TimeCompatibleProtocol):
        return self.transpose()

    def moveaxis(self: TimeCompatibleProtocol, source: int | Tuple[int], destination: int | Tuple[int]):
        if isinstance(source, int):
            source = (source,)
        if isinstance(destination, int):
            destination = (destination,)

        cls = self._get_array_cls()

        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        moved_array: TimeCompatibleProtocol = np.moveaxis(np.asarray(self), source, destination).view(
            cls
        )  # type: ignore
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

    def rollaxis(self: TimeCompatibleProtocol, axis: int, start: int = 0):
        # we re-instanciate a TimelinedArray with view instead of the full constructor : faster
        cls = self._get_array_cls()

        rolled_array: TimeCompatibleProtocol = np.rollaxis(np.asarray(self), axis, start).view(cls)  # type: ignore
        rolled_array.timeline = self.timeline
        rolled_array.time_dimension = self.time_dimension

        if axis < self.time_dimension:
            if start <= axis:
                rolled_array.time_dimension += 1
            elif start <= self.time_dimension:
                rolled_array.time_dimension -= 1
        elif axis == self.time_dimension:
            rolled_array.time_dimension = start
        else:
            if start <= self.time_dimension:
                rolled_array.time_dimension -= 1
            elif start > self.time_dimension:
                rolled_array.time_dimension += 1

        # TimelinedArray.time_dimension and TimelinedArray.timeline are set. good to go
        return rolled_array

    def mean(self: TimeCompatibleProtocol, axis: int | Tuple[int] | None = None, dtype=None, out=None, keepdims=False):
        result = super().mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        return self._finish_axis_removing_operation(result, axis)

    # Override other reduction methods similarly if needed
    def sum(self: TimeCompatibleProtocol, axis: int | Tuple[int] | None = None, dtype=None, out=None, keepdims=False):
        result = super().sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        return self._finish_axis_removing_operation(result, axis)

    def std(
        self: TimeCompatibleProtocol, axis: int | Tuple[int] | None = None, dtype=None, out=None, ddof=0, keepdims=False
    ):
        result = super().std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        return self._finish_axis_removing_operation(result, axis)

    def var(
        self: TimeCompatibleProtocol, axis: int | Tuple[int] | None = None, dtype=None, out=None, ddof=0, keepdims=False
    ):
        result = super().var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        return self._finish_axis_removing_operation(result, axis)

    def rebase_timeline(self, at=0):
        # returns a modified version of the array, with the first element of the array to time zero,
        # and shift the rest accordingly
        cls = self._get_array_cls()
        return cls(self, timeline=self.timeline - self.timeline[at])  # type: ignore

    def offset_timeline(self, offset):
        # returns a modified version of the array, where we set time of all elements
        # in array at a fix offset relative to their current value.
        cls = self._get_array_cls()
        return cls(self, timeline=self.timeline + offset)  # type: ignore

    @property
    def pack(self):
        return TimePacker(self)

    def sec_max(self):
        # get maximum time
        return self.timeline.max()

    def sec_min(self):
        # get minimum time
        return self.timeline.min()

    @staticmethod
    def extract_time_from_data(data, timeline=None, time_dimension=None, uniform_space=False):
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
                time_dimension = getattr(element, "time_dimension", None) + 1
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
        return iter((self.array.timeline, self.array.__array__()))


class TimelinedArray(TimeMixin, np.ndarray, TimeCompatibleProtocol):
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
        seconds_to_index: This method converts time in seconds to index value.
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

    def __new__(cls, data, timeline=None, time_dimension: int | None = None, uniform_space=False) -> "TimelinedArray":

        data, timeline, time_dimension = TimeMixin.extract_time_from_data(
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
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.timeline = getattr(obj, "timeline", Timeline([]))
        self.time_dimension = getattr(obj, "time_dimension", 0)

    def __array_wrap__(self, out_arr, context=None):
        if context is not None:
            logger.debug(f"wrapping array after ufunc {context[0].__name__}")
        output = super().__array_wrap__(out_arr, context)
        if len(output.shape) < len(self.shape):
            logger.debug(f"shape reduced from : {self.shape} to : {output.shape}. outarray was : {out_arr.shape}")
        return output

    def __array_function__(self, func, types, args, kwargs):
        logger.debug(f"intercepting array before function {func.__name__}")
        return super().__array_function__(func, types, args, kwargs)

    def __getitem__(
        self, index: int | Tuple[int] | slice | Tuple[slice] | List | np.ndarray
    ) -> "TimelinedArray | np.ndarray":

        index, final_timeline, final_time_dimension = self._get_indexed_times(index)

        if final_timeline is None or final_time_dimension is None:
            return np.array(self).__getitem__(index)

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
        return type(self).__name__ + np.array(self).__repr__()[5:]

    def __str__(self):
        return type(self).__name__ + np.array(self).__str__()

    @staticmethod
    def align_from_iterable(iterable) -> "TimelinedArray":
        start = min([item.timeline.min() for item in iterable])
        maxlen = min([len(item.isec[start:]) for item in iterable])

        aligned_arrays = []
        for index, item in enumerate(iterable):
            aligned_arrays.append(item.align_trace(start, maxlen))

        return TimelinedArray(aligned_arrays)


class MaskedTimelinedArray(TimeMixin, np.ma.MaskedArray, TimeCompatibleProtocol):
    def __new__(
        cls,
        data,
        mask=np.ma.nomask,
        dtype=None,
        copy=False,
        fill_value=None,
        keep_mask=True,
        hard_mask=False,
        shrink=True,
        timeline=None,
        time_dimension=None,
        uniform_space=False,
        **kwargs,
    ):

        _, timeline, time_dimension = TimeMixin.extract_time_from_data(
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
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.timeline = getattr(obj, "timeline", Timeline([]))
        self.time_dimension = getattr(obj, "time_dimension", 0)

    def __getitem__(
        self, index: int | Tuple[int] | slice | Tuple[slice] | List | np.ndarray
    ) -> "MaskedTimelinedArray | np.ma.MaskedArray":

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
