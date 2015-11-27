import numpy as np
import matplotlib.pyplot as plt


class Erds(object):
    """ERDS maps

    Parameters
    ----------
    TODO

    Attributes
    ----------
    erds_ : array, shape (n_fft, n_channels, n_segments)

    Examples
    --------
    TODO
    """
    def __init__(self, n_fft=128, n_segments=32, baseline=None, fs=None):
        self.n_fft = n_fft  # frequency bins in ERDS map
        self.n_segments = n_segments  # number of time points in ERDS map
        self.baseline = baseline  # baseline interval
        self.fs = fs  # sampling frequency

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
            baseline = np.asarray(self.baseline)  # TODO: baseline should be provided in samples? or seconds? now it's in segments.

        e, c, t = epochs.shape
        self.erds_ = []

        stft = []
        for epoch in range(e):
            stft.append(self._stft(epochs[epoch, :, :]))
        stft = np.stack(stft, axis=-1).mean(axis=-1)

        ref = stft[baseline[0]:baseline[1] + 1, :, :].mean(axis=0)
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
        pad = np.zeros((c, self.n_fft / 2))  # TODO: is this correct for odd self.n_fft?
        x = np.concatenate((pad, x, pad), axis=-1)  # zero-pad
        step = t // (self.n_segments - 1)
        stft = np.empty((self.n_segments, c, self.n_fft // 2 + 1))
        window = np.hanning(self.n_fft)

        for segment in range(self.n_segments):
            start = segment * step
            end = start + self.n_fft
            windowed = x[:, start:end] * window
            spectrum = np.fft.rfft(windowed) / self.n_fft
            stft[segment, :, :] = np.abs(spectrum * np.conj(spectrum))
        return stft

    def plot(self, channels=None, f=None):
        """Plot ERDS maps.
        """
        # TODO: plot specified channels
        plt.imshow(self.erds_[:60, 0, :], origin="lower", aspect="auto",
                   interpolation="none", cmap=plt.get_cmap("jet_r"))
