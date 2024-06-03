import unittest
import numpy as np
from numpy.testing import assert_array_equal
from timelined_array.time import TimeIndexer, Timeline, TimelinedArray, MaskedTimelinedArray


class TestTimeline(unittest.TestCase):
    def test_timeline_creation(self):
        timeline = Timeline([0, 1, 2, 3, 4])
        self.assertEqual(timeline.max(), 4)
        self.assertEqual(timeline.min(), 0)

    def test_timeline_uniformize(self):
        with self.assertRaises(NotImplementedError):
            Timeline._uniformize([0, 1, 2, 3, 4])

    def test_timeline_step(self):
        timeline = Timeline([0, 1, 2, 3, 4])
        self.assertEqual(timeline.step, 1)

    def test_timeline_max_step(self):
        timeline = Timeline([0, 1, 2, 3, 4])
        self.assertEqual(timeline.max_step, 2)


class TestTimeIndexer(unittest.TestCase):
    def setUp(self):
        self.timeline = Timeline([0, 1, 2, 3, 4])
        self.array = TimelinedArray(np.random.rand(5, 5), timeline=self.timeline, time_dimension=0)
        self.indexer = TimeIndexer(self.array)

    def test_time_to_index(self):
        self.assertEqual(self.indexer.time_to_index(2), 2)

    def test_get_iindex(self):
        self.assertEqual(self.indexer.get_iindex(1, 3), slice(1, 3, 1))


class TestTimelinedArray(unittest.TestCase):
    def setUp(self):
        self.timeline = Timeline([0, 1, 2, 3, 4])
        self.array = TimelinedArray(np.random.rand(5, 5), timeline=self.timeline, time_dimension=0)

    def test_array_creation(self):
        self.assertEqual(self.array.shape, (5, 5))
        self.assertEqual(self.array.timeline.max(), 4)
        self.assertEqual(self.array.time_dimension, 0)

    def test_array_indexing(self):
        indexed_array = self.array[1:3]
        self.assertEqual(indexed_array.shape, (2, 5))
        self.assertEqual(indexed_array.timeline.min(), 1)
        self.assertEqual(indexed_array.timeline.max(), 2)

    def test_array_transpose(self):
        transposed_array = self.array.transpose()
        self.assertEqual(transposed_array.shape, (5, 5))
        self.assertEqual(transposed_array.time_dimension, 1)

    def test_array_mean(self):
        mean_value = self.array.mean(axis=0)
        self.assertEqual(mean_value.shape, (5,))


class TestMaskedTimelinedArray(unittest.TestCase):
    def setUp(self):
        self.timeline = Timeline([0, 1, 2, 3, 4])
        self.data = np.random.rand(5, 5)
        self.mask = np.zeros((5, 5), dtype=bool)
        self.mask[0, 0] = True
        self.array = MaskedTimelinedArray(self.data, mask=self.mask, timeline=self.timeline, time_dimension=0)

    def test_masked_array_creation(self):
        self.assertEqual(self.array.shape, (5, 5))
        self.assertEqual(self.array.timeline.max(), 4)
        self.assertEqual(self.array.time_dimension, 0)
        self.assertTrue(self.array.mask[0, 0])

    def test_masked_array_indexing(self):
        indexed_array = self.array[1:3]
        self.assertEqual(indexed_array.shape, (2, 5))
        self.assertEqual(indexed_array.timeline.min(), 1)
        self.assertEqual(indexed_array.timeline.max(), 2)


class TestMultidimensionnalIndexing(unittest.TestCase):
    def setUp(self):
        dimension = (15, 150, 200, 200)
        y_dim_scale = 2
        prime = [
            [
                (
                    np.array(
                        [
                            np.roll(np.linspace(phase, phase * 2, num=dimension[3], endpoint=True), shift=r)
                            for r in np.arange(0, dimension[2] * y_dim_scale, y_dim_scale)
                        ]
                    )
                    * t
                )
                for t in np.sin(np.linspace(phase, phase + 2, num=dimension[1]))
            ]
            for phase in np.linspace(1, dimension[0], num=dimension[0], endpoint=True)
        ]
        prime = np.array(prime)
        sin = np.sin(prime)
        self.array = TimelinedArray(sin, time_dimension=1, timeline=np.linspace(-2, 7, num=300))


if __name__ == "__main__":
    unittest.main()
