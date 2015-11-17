import numpy as np
import matplotlib.pyplot as plt


class Erds(object):
    def __init__(self, baseline=None):
        self.baseline = baseline  # None means whole epoch
        self.nfft = 128  # frequency resolution
        self.nsegments = 32  # number of time points in ERDS map

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

        stft = []
        for epoch in range(e):
            stft.append(self._stft(epochs[epoch, :, :]))
        stft = np.stack(stft, axis=-1).mean(axis=-1)

        ref = stft[baseline[0]:baseline[1], :, :].mean(axis=0)
        # split into list of channels
        self.erds_ = [ch.squeeze().T for ch in np.split(stft / ref - 1, c, 1)]

        return self

    def _stft(self, x):
        """Compute Short-Time Fourier Transform (STFT).

        Parameters
        ----------
        x : array, shape (n_channels, n_times)
            Data used to compute STFT.

        Returns
        -------
        stft : array, shape (n_segments, n_channels, n_freqs)
            STFT of x.
        """
        c, t = x.shape
        pad = np.zeros((c, self.nfft / 2))  # self.nfft must be a power of 2
        x = np.concatenate((pad, x, pad), axis=-1)  # zero-pad
        step = int(t / (self.nsegments - 1))
        stft = np.empty((self.nsegments, c, self.nfft))
        window = np.hanning(self.nfft)

        for k in range(self.nsegments):
            start = k * step
            end = start + self.nfft
            windowed = x[:, start:end] * window
            spectrum = np.fft.fft(windowed) / self.nfft
            stft[k, :, :] = np.abs(spectrum * np.conj(spectrum))
        return stft

    def plot(self):
        """Plot ERDS maps.
        """
        pass


a = np.random.randn(100, 64, 1000)
erds = Erds()
erds.fit(a)
plt.imshow(erds.erds_[0], origin="lower", aspect="auto", interpolation="none")
plt.show()
