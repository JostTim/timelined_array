import numpy as np
import pytest

from timelined_array import TimelinedArray

AX0_SIZE_3D = 20
AX1_SIZE_3D = 50
AX2_SIZE_3D = 75

TIME_AXIS_3D = 1
TIME_AXIS_SIZE_3D = locals()[f"AX{TIME_AXIS_3D}_SIZE_3D"]
ARRAY_SHAPE_3D = (AX0_SIZE_3D, AX1_SIZE_3D, AX2_SIZE_3D)

AXIS_3D_MAPPING = {0: AX0_SIZE_3D, 1: AX1_SIZE_3D, 2: AX2_SIZE_3D}


@pytest.fixture
def timelined_array_3D():
    return TimelinedArray(
        np.random.rand(*ARRAY_SHAPE_3D),
        timeline=np.arange(TIME_AXIS_SIZE_3D),
        time_dimension=TIME_AXIS_3D,
    )


TIME_AXIS_1D = 0
TIME_AXIS_SIZE_1D = 50
ARRAY_SHAPE_1D = (TIME_AXIS_SIZE_1D,)


@pytest.fixture
def timelined_array_1D():
    return TimelinedArray(
        np.random.rand(TIME_AXIS_SIZE_1D),
        timeline=np.arange(TIME_AXIS_SIZE_1D),
        time_dimension=TIME_AXIS_1D,
    )
