import numpy as np
import pytest
from timelined_array import TimelinedArray, MaskedTimelinedArray
from timelined_array.time import (Timeline,
                                  StartBoundary,
                                  StoptBoundary,
                                  EdgePolicy,
                                  TimeIndexer,
                                  TimeMixin,
                                  TimePacker)


ARRAY_1D_SHAPE = (50,)
ARRAY_3D_SHAPE = (25, 50, 75)
ARRAY_3D_TIME_DIMENSION = 1


@pytest.fixture
def timelined_array_1D():
    return TimelinedArray(np.random.rand(*ARRAY_1D_SHAPE),
                          timeline=np.arange(ARRAY_1D_SHAPE[0]),
                          time_dimension=0)


@pytest.fixture
def timelined_array_3D():
    return TimelinedArray(np.random.rand(*ARRAY_3D_SHAPE),
                          timeline=np.arange(
                              ARRAY_3D_SHAPE[ARRAY_3D_TIME_DIMENSION]),
                          time_dimension=ARRAY_3D_TIME_DIMENSION)


@pytest.mark.parametrize("shape, timeline_dimension", [((12,), (10, 30, 50), (29, 34, 61, 15, 4)), (0, 2, 0)])
def test_ta_array_shape(shape, timeline_dimension):
    ta_array = TimelinedArray(np.random.rand(*shape),
                              timeline=np.arange(
                              shape[timeline_dimension]),
                              time_dimension=timeline_dimension)

    assert ta_array.shape == shape
