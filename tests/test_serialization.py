import pickle
from pathlib import Path

import numpy as np
import pytest

from timelined_array import TimelinedArray


@pytest.fixture
def pickle_path(tmp_path: Path):
    file_path = tmp_path / "serialized_test_3D_ta_array.pickle"
    yield file_path


@pytest.mark.filterwarnings(
    "ignore:FutureWarning: We do not yet support timeline management"
)
def test_pickle_unpickle(timelined_array_3D: TimelinedArray, pickle_path):
    # Pickle the numpy array
    with open(pickle_path, "wb") as f:
        pickle.dump(timelined_array_3D, f)

    # Unpickle the numpy array
    with open(pickle_path, "rb") as f:
        unserialized_3D_array = pickle.load(f)

    # Assert that the original and unpickled arrays are the same
    np.testing.assert_array_equal(timelined_array_3D, unserialized_3D_array)
    np.testing.assert_array_equal(
        timelined_array_3D.timeline, unserialized_3D_array.timeline
    )
    assert timelined_array_3D.time_dimension == unserialized_3D_array.time_dimension


def test_representation(timelined_array_3D: TimelinedArray):

    assert str(timelined_array_3D).startswith("TimelinedArray")
    assert timelined_array_3D.__repr__().startswith("TimelinedArray")
