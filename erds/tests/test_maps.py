import unittest
import numpy as np
import erds


class TestMaps(unittest.TestCase):
    def test_dimensions(self):
        testdata = np.random.randn(100, 64, 1000)
        n_fft = 128
        n_segments = 32
        e, c, t = testdata.shape
        maps = erds.Erds(n_fft=n_fft, n_segments=n_segments)
        maps.fit(testdata)
        self.assertEqual(maps.erds_.shape, (n_fft, c, n_segments))


if __name__ == '__main__':
    unittest.main()
