import numpy as np
import matplotlib.pyplot as plt


class Erds(object):
    def __init__(self, baseline=None):
        self.baseline = baseline  # None means whole epoch
        self.nfft = 256  # frequency resolution
        self.overlap = 128  # time resolution

    def fit(self, epochs):
        """Compute ERDS maps.

        Parameters
        ----------
        epochs : array, shape (n_epochs, n_channels, n_times)
            Data used to compute ERDS maps.

        Returns
        -------
        self: instance of Erds
            Returns the modified instance.
        """
        X = epochs.mean(axis=0)
        c, t = X.shape
        self.erds_ = []
        if self.baseline is None:
            baseline = 0, t  # TODO: baseline cannot range from 0-t, but has
                             # indices from 0 to n_segments (of STFT)

        for channel in range(c):
            stft = self._stft(X[channel, :])
            ref = stft[baseline[0]:baseline[1], :].mean(axis=0)
            self.erds_.append(np.transpose(stft / ref - 1))

        return self

    def _stft(self, x):
        """Compute Short-Time Fourier Transform (STFT) for single channel.

        Parameters
        ----------
        x : array, shape (n_times,)
            Data used to compute STFT (single channel).

        Returns
        -------
        stft : array, shape (n_segments, n_freqs)
            STFT of x.
        """

        window = np.hanning(self.nfft)
        hop = self.nfft - self.overlap
        n_segments = int(np.ceil(len(x) / hop))
        stft = np.empty((n_segments, self.nfft))
        x = np.concatenate((x, np.zeros((self.nfft))))

        for k in range(n_segments):
            start = k * hop
            end = start + self.nfft
            windowed = x[start:end] * window
            spectrum = np.fft.fft(windowed) / self.nfft
            stft[k, :] = np.abs(spectrum * np.conj(spectrum))
        return stft

    def plot(self):
        """Plot ERDS maps.
        """
        pass


a = np.random.randn(100, 64, 2048)
erds = Erds()
erds.fit(a)
plt.imshow(erds.erds_[0], origin="lower", aspect="auto")
plt.show()
