import numpy as np
import matplotlib.pyplot as plt


class Erds(object):
    """ERDS maps

    Parameters
    ----------
    TODO

    Attributes
    ----------
    erds_ : array, shape (n_segments, n_channels, n_fft)

    Examples
    --------
    TODO
    """
    def __init__(self, baseline=None):
        self.n_fft = 128  # frequency resolution
        self.n_segments = 32  # number of time points in ERDS map
        self.baseline = baseline

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
        if self.baseline is None:
            baseline = np.array([0, self.n_segments])
        else:
            baseline = np.asarray(self.baseline) // self.n_segments

        e, c, t = epochs.shape
        self.erds_ = []

        stft = []
        for epoch in range(e):
            stft.append(self._stft(epochs[epoch, :, :]))
        stft = np.stack(stft, axis=-1).mean(axis=-1)

        ref = stft[baseline[0]:baseline[1] + 1, :, :].mean(axis=0)
        # split into list of channels
        self.erds_ = (stft / ref - 1).transpose(2, 1, 0)

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
        pad = np.zeros((c, self.n_fft / 2))  # self.nfft must be a power of 2
        x = np.concatenate((pad, x, pad), axis=-1)  # zero-pad
        step = t // (self.n_segments - 1)
        stft = np.empty((self.n_segments, c, self.n_fft))
        window = np.hanning(self.n_fft)

        for k in range(self.n_segments):
            start = k * step
            end = start + self.n_fft
            windowed = x[:, start:end] * window
            spectrum = np.fft.fft(windowed) / self.n_fft
            stft[k, :, :] = np.abs(spectrum * np.conj(spectrum))
        return stft

    def plot(self, channels=None):
        """Plot ERDS maps.
        """
        # TODO: plot specified channels
        plt.imshow(self.erds_[:, 0, :], origin="lower", aspect="auto",
                   interpolation="none")


a = np.random.randn(100, 64, 1000)
erds = Erds()
erds.fit(a)
erds.plot()
plt.show()
