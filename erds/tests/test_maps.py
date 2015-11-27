import unittest
import numpy as np
import erds


class TestMaps(unittest.TestCase):
    def test_dimensions(self):
        testdata = np.random.randn(100, 64, 1000)  # epochs, channels, samples
        n_times = 32
        n_freqs = 128
        e, c, t = testdata.shape
        maps = erds.Erds(n_times=n_times, n_freqs=n_freqs)
        maps.fit(testdata)
        self.assertEqual(maps.erds_.shape, (n_freqs, c, n_times))


if __name__ == '__main__':
    unittest.main()
