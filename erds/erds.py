import numpy as np
import matplotlib.pyplot as plt


class Erds(object):
    """ERDS maps

    Parameters
    ----------
    TODO

    Attributes
    ----------
    erds_ : array, shape (n_freqs, n_channels, n_times)

    Examples
    --------
    TODO
    """
    def __init__(self, n_times=128, n_freqs=513, baseline=None, fs=1):
        self.n_times = n_times  # number of time points in ERDS map
        self.n_freqs = n_freqs  # number of frequency bins in ERDS map
        self.baseline = baseline  # baseline interval start and end
        self.fs = fs  # sampling frequency

    def fit(self, epochs):
        """Compute ERDS maps.

        Parameters
        ----------
        epochs : array, shape (n_epochs, n_channels, n_samples)
            Data used to compute ERDS maps.

        Returns
        -------
        self: instance of Erds
            Returns the modified instance.
        """
        e, c, t = epochs.shape
        self.n_epochs_ = e
        self.n_channels_ = c
        self.n_samples_ = t
        self.erds_ = []
        self.midpoints_ = np.arange(0, t, t // (self.n_times - 1)) / self.fs
        self.n_fft_ = (self.n_freqs - 1) * 2

        # if self.fs is not None:
        #     self.freqs_ = np.fft.rfftfreq(self.n_fft_) * self.fs

        if self.baseline is None:  # use whole epoch
            self.baseline_ = np.arange(0, self.n_times)
        else:
            # find corresponding closest times
            tmp = [np.abs(self.midpoints_ - v).argmin() for v in self.baseline]
            self.baseline_ = np.arange(*tmp)

        stft = []
        for epoch in range(e):
            stft.append(self._stft(epochs[epoch, :, :]))
        stft = np.stack(stft, axis=-1).mean(axis=-1)

        ref = stft[self.baseline_, :, :].mean(axis=0)
        self.erds_ = (stft / ref - 1).transpose(2, 1, 0)

        return self

    def _stft(self, x):
        """Compute Short-Time Fourier Transform (STFT).

        Parameters
        ----------
        x : array, shape (n_channels, n_samples)
            Data used to compute STFT.

        Returns
        -------
        stft : array, shape (n_times, n_channels, n_freqs)
            STFT of x.
        """
        c, t = x.shape
        pad = np.zeros((c, self.n_fft_ // 2))
        x = np.concatenate((pad, x, pad), axis=-1)  # zero-pad
        step = t // (self.n_times - 1)
        stft = np.empty((self.n_times, c, self.n_freqs))
        window = np.hanning(self.n_fft_)

        for time in range(self.n_times):
            start = time * step
            end = start + self.n_fft_
            windowed = x[:, start:end] * window
            spectrum = np.fft.rfft(windowed) / self.n_fft_
            stft[time, :, :] = np.abs(spectrum * np.conj(spectrum))
        return stft

    def plot(self, channels=None, f_min=0, f_max=None, t_min=0, t_max=None,
             nrows=None, ncols=None):
        """Plot ERDS maps.

        Parameters
        ----------
        channels : array or None
            Channels to display. If None, ERDS maps for all channels are
            displayed.
        """
        # TODO: plot specified channels
        if f_max is None:
            f_max = self.fs / 2
        c = self.n_freqs / (self.fs / 2)

        nrows = np.ceil(np.sqrt(self.n_channels_)) if nrows is None else nrows
        ncols = np.ceil(np.sqrt(self.n_channels_)) if ncols is None else ncols

        for ch in range(self.n_channels_):
            plt.subplot(nrows, ncols, ch + 1)
            plt.imshow(self.erds_[f_min * c:f_max * c, ch, :], origin="lower",
                       aspect="auto", interpolation="none",
                       cmap=plt.get_cmap("jet_r"),
                       extent=[0, self.n_samples_ / self.fs, f_min, f_max])
            plt.title(str(ch + 1))
        plt.tight_layout(1)
        # TODO: return figure?
