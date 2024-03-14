# -*- coding: utf-8 -*-

import numpy as np

# TODO : do numpy.interp(np.arange(0, len(a), 1.5), np.arange(0, len(a)), a) interp to get a fixed number of points ?


class TimelinedArray(np.ndarray):
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

    time_dimension: int
    timeline: np.ndarray

    class TA_Timeline(np.ndarray):
        def __new__(cls, input_array, uniform_space=False):

            if uniform_space:
                # if we want a uniformly spaced timeline from start to stop of the current timeline.
                obj = np.linspace(input_array[0], input_array[1], len(input_array)).view(cls)

            else:
                if isinstance(input_array, TimelinedArray.TA_Timeline):
                    return input_array
                obj = np.asarray(input_array).view(cls)

            return obj

        def __array_finalize__(self, obj):
            pass

        def __setstate__(self, state):
            try:
                super().__setstate__(state[0:-2])  # old deserializer
            except TypeError:
                super().__setstate__(state)  # new one

        def __contains__(self, time_value):
            if time_value >= self.min() and time_value <= self.max():
                return True
            return False

        def max(self):
            return super().max().item()

        def min(self):
            return super().min().item()

    class TA_Isec_Indexer:

        def __init__(self, array):
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

        def __getitem__(self, index):
            if hasattr(index, "__iter__"):
                # if not isinstance(index,(int,float,slice,np.integer,np.floating)):
                raise ValueError(
                    "Isec allow only indexing on time dimension. Index must be either int, float or slice, not iterable"
                )

            iindex_time = self.seconds_to_index(index)
            full_iindex = self._insert_time_index(iindex_time)
            # print("new full index : ",iindex_time)
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

    class TA_Packer:
        def __init__(self, array):
            self.array = array

        def __iter__(self):
            return iter((self.array.timeline, self.array.__array__()))

    def __new__(cls, input_array, timeline=None, time_dimension: int | None = None, uniform_space=False):

        _unpacking = False
        if (
            timeline is None
        ):  # if timeline not explicitely passed as arg, we try to pick up the timeline of the input_array.
            # will rise after if input_array is not a timelined_array
            try:
                timeline = input_array.timeline
            except AttributeError:
                try:  # if arguments are an uniform list of timelined array
                    # (often use to make mean and std of synchonized timelines), we pick up the first one.
                    for element in input_array:
                        timeline = element.timeline
                        _unpacking = True
                        break
                except AttributeError:
                    raise ValueError("timeline must be supplied if the input_array is not a TimelinedArray")

        if time_dimension is None:  # same thing for the time dimension.
            try:
                time_dimension = int(input_array.time_dimension)
            except AttributeError:
                try:  # if arguments are an uniform list of timelined array
                    # (often use to make mean and std of synchonized timelines), we pick up the first one.
                    # but it also means default numpy packing will set the new dimension as dimension 0.
                    # As such, the current time dimension will have to be the time dimension of the listed elements,
                    # +1 (a.k.a. shifted one dimension deeper)
                    for element in input_array:
                        time_dimension = int(element.time_dimension) + 1
                        _unpacking = True
                        break
                    else:
                        time_dimension = 0
                except AttributeError:
                    time_dimension = 0

        timeline = TimelinedArray.TA_Timeline(timeline, uniform_space=uniform_space)

        if _unpacking and len(input_array.shape) <= time_dimension:  # type: ignore
            input_array = np.stack(input_array)

        obj = np.asarray(input_array).view(
            cls
        )  # instanciate the np array as a view, as per numpy documentation on how to make ndarray child classes

        if obj.shape[time_dimension] != len(timeline):
            raise ValueError(
                "timeline object and the shape of time_dimension of the input_array must be equal. "
                f"They are : {len(timeline)} and {obj.shape[time_dimension]}"
            )

        obj.timeline = timeline
        obj.time_dimension = time_dimension
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # else, we can reassign attributes of the old array, after a transformation for example .mean, etc
        # TODO : change here to include safechecks to change the timeline if the array changed.
        # maybe using original_shape as a memo of the shape before transormation,
        # to find out wich dimension was reduced ?
        self.timeline = getattr(obj, "timeline", np.array([]))
        self.time_dimension = getattr(obj, "time_dimension", 0)

    def rebase_timeline(self):
        # returns a modified version of the array, with the first element of the array to time zero,
        # and shift the rest accordingly
        return TimelinedArray(self, timeline=self.timeline - self.timeline[0])

    def offset_timeline(self, offset):
        # returns a modified version of the array, where we set time of all elements
        # in array at a fix offset relative to their current value.
        return TimelinedArray(self, timeline=self.timeline + offset)

    @property
    def pack(self):
        return TimelinedArray.TA_Packer(self)

    def __getitem__(self, index):
        # we re-implement the index getting method of numpy, to crop the timeline with the array,
        # when the array is asked to be cropped.

        # first we must check wether the time dimension has evolved (less/more dimensions)
        # if index is not iterable, then we affect only dimension 0
        # we define a placeholder as a tuple just for code flexibility to all cases

        # just in case index is a numpy array with one element, we get it as a single element and not an array
        # as an array has __iter__ but a single element array has no __len__, so the next code would break
        if isinstance(index, np.ndarray):
            if index.size == 1:
                index = index.item()

        _index = index if hasattr(index, "__iter__") else (index,)
        _time_dimension = self.time_dimension
        _time_dimension_in_index = self.time_dimension

        for dimension in range(len(_index)):
            if dimension > _time_dimension_in_index:
                continue  # if the _index[dimension] changes above time dimension,
                # we don't care as it will not affect it's position.
            if _index[dimension] is None:  # np.newaxis is a placeholder for None. a None in an index == a new axis.
                # in that case, it means a dimension was added before time_dimension, so we will shift it by one.
                _time_dimension += 1
                _time_dimension_in_index += (
                    1  # we also will look at values for time dimension here to apply to timeline later
                )
            elif isinstance(_index[dimension], (int, np.integer)):
                if dimension == _time_dimension:
                    # if the time dimension IS an integer, the use will loose time anyway
                    # (getting a single point in time)
                    # so in that case, we stop further thinking in term of TimelinedArray and return a standard array,
                    # indexed as wanted.
                    return np.array(self).__getitem__(index)
                # otherwise it is below, so in that case, if one dimension is removed,
                # bu selecting an single value or that dimension, we substract one to time dimension.
                _time_dimension -= 1
                # here we don't remove one to _time_dimension_in_index.
                # #This -1 removal is for the new array, but the current array still has some somension here that will
                # be indexed,and we need to know that value for timeline.

        # (index, _time_dimension, _time_dimension_in_index)

        if len(_index) >= _time_dimension_in_index + 1:
            # if a part of the index was destined to reshape the time dimension,
            # we apply this reshaping to timeline too.
            _timeline = self.timeline[_index[_time_dimension_in_index]]
        else:
            _timeline = self.timeline  # if time_dimension is not part of index, then timeline doesn't change.

        return TimelinedArray(super().__getitem__(index), timeline=_timeline, time_dimension=_time_dimension)

    @property
    def isec(self):
        """Allows to use my_array.isec[12.5] to get element closes to time 12.5 in the array's timeline.
        Works with spans of time, and uses item based index for dimensions that are not a time_dimension

        Returns:
            TA_Isec_Indexer: the objects that can index on the TimelinedArray,
                and returns a TimelinedArray or standard nparray upon __getitem__ call.
                (Standard ndarray array if the time dimension collapses to size 1)
        """
        return TimelinedArray.TA_Isec_Indexer(self)

    def sec_max(self):
        # get maximum time
        return self.timeline.max()

    def sec_min(self):
        # get minimum time
        return self.timeline.min()

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.timeline, self.time_dimension)  # type: ignore

        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        self.timeline = state[-2]  # Set the info attribute
        self.time_dimension = state[-1]
        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-2])

    def __hash__(self):
        return hash((self.__array__(), self.timeline))

    # THESE ARE OVERRIDEN TO AVOID HORRIBLE PERFORMANCE WHEN PRINTING
    # DUE TO CUSTOM __GETITEM__ PRE-CHECKS WITH RECURSIVE NATIVE NUMPY REPR
    def __repr__(self):
        return type(self).__name__ + np.array(self).__repr__()[5:]

    def __str__(self):
        return type(self).__name__ + np.array(self).__str__()

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

        return self.isec[start:][:element_nb]
