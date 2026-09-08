import numpy as np
import pytest

from timelined_array import Timeline


def test_timeline_creation():
    timeline = Timeline(np.arange(10))
    assert isinstance(timeline, Timeline)
    assert timeline.max() == 9
    assert timeline.min() == 0
    assert 5 in timeline
    assert timeline.step == 1
    assert timeline.is_uniform

    with pytest.raises(NotImplementedError):
        timeline.uniformize()

    non_uniform_timeline = Timeline([0, 1, 2, 4, 3, 1.5, 6, 7, 0.5])

    with pytest.raises(ValueError):
        non_uniform_timeline.step
        assert True
