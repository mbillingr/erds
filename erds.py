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
        e, c, t = epochs.shape
        self.erds_ = []
        if self.baseline is None:
            baseline = 0, t

        for channel in range(c):
            stft = []
            for epoch in range(e):
                stft.append(self._stft(epochs[epoch, channel, :]))
            stft = np.stack(stft, axis=-1).mean(axis=-1)

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
        x = np.concatenate((x, np.zeros((self.nfft))))  # zero-pad

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


a = np.random.randn(100, 64, 1024)
erds = Erds()
erds.fit(a)
plt.imshow(erds.erds_[0], origin="lower", aspect="auto", interpolation="none")
plt.show()
