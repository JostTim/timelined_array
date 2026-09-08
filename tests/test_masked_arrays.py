import numpy as np
import pytest

from timelined_array import MaskedTimelinedArray


@pytest.fixture
def masked_timelined_array_1D():
    data = np.random.rand(50)
    mask = data > 0.5
    return MaskedTimelinedArray(
        data, mask=mask, timeline=np.arange(50), time_dimension=0
    )


def test_masked_ta_creation():
    data = np.random.rand(10, 10)
    mask = data > 0.5
    masked_ta = MaskedTimelinedArray(
        data, mask=mask, timeline=np.arange(10), time_dimension=0
    )
    assert isinstance(masked_ta, MaskedTimelinedArray)
    assert masked_ta.shape == (10, 10)
    assert masked_ta.time_dimension == 0


def test_masked_timelined_array_indexing(
    masked_timelined_array_1D: MaskedTimelinedArray,
):
    sub_array = masked_timelined_array_1D[10:20]
    assert isinstance(sub_array, MaskedTimelinedArray)
    assert sub_array.shape == (10,)
    assert np.array_equal(sub_array.timeline, np.arange(10, 20))
    assert np.all(sub_array.mask == (sub_array.data > 0.5))


def test_masked_timelined_array_1D_properties(
    masked_timelined_array_1D: MaskedTimelinedArray,
):
    assert masked_timelined_array_1D.time_dimension == 0
    assert np.all(
        masked_timelined_array_1D.mask == (masked_timelined_array_1D.data > 0.5)
    )
    assert masked_timelined_array_1D.timeline.shape == (50,)
