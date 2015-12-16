import numpy as np
import matplotlib.pyplot as plt


def bootstrap(x, n_resamples=5000, statistic="mean", alpha=0.01):
    """Compute bootstrap confidence interval estimate of statistic.

    Parameters
    ----------
    x : array, shape (n, )
        One-dimensional input data.
    n_resamples : int
        Number of bootstrap resamples to draw.
    statistic : str or function
        Statistic for which bootstrap confidence intervals are computed.
    alpha : float
        Significance level.

    Returns
    -------
    cl, cu : tuple
        Lower and upper confidence interval boundaries.
    """
    # each bootstrap sample has the same length as the input data
    samples = np.random.choice(x, (n_resamples, len(x)))
    if statistic == "mean":
        statistic = np.mean
    elif statistic == "median":
        statistic = np.median
    stat = np.sort(statistic(samples, 1))
    return (stat[int(alpha/2 * n_resamples)],
            stat[int((1 - alpha/2) * n_resamples)])


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
        self.erds_ = None

    def __repr__(self):
        s =  "<Erds object>\n"
        s += "  n_times: {}\n".format(self.n_times)
        s += "  n_freqs: {}\n".format(self.n_freqs)
        if self.baseline is None:
            s += "  baseline: whole epoch\n".format(self.baseline)
        else:
            s += "  baseline: {} s\n".format(self.baseline)
        s += "  fs: {} Hz\n".format(self.fs)
        if self.erds_ is None:
            s += "  ERDS maps have not been computed (use fit method).\n"
        else:
            s += "  Input data:\n"
            s += "    epochs: {}\n".format(self.n_epochs_)
            s += "    channels: {}\n".format(self.n_channels_)
            s += "    length: {} samples\n".format(self.n_samples_)
            s += "  ERDS data:\n"
            s += "    frequency bins: {}\n".format(self.n_freqs)
            s += "    channels: {}\n".format(self.n_channels_)
            s += "    length: {} samples\n".format(self.n_times)

        return s

    def fit(self, epochs, fs=None, sig=False):
        """Compute ERDS maps.

        Parameters
        ----------
        epochs : array, shape (n_epochs, n_channels, n_samples)
            Data used to compute ERDS maps.
        fs : float
            Sampling frequency.
        sig : bool
            Whether or not to calculate significance mask for ERDS maps.

        Returns
        -------
        self: instance of Erds
            Returns the modified instance.
        """
        if fs is not None:
            self.fs = fs
        e, c, t = epochs.shape
        self.n_epochs_ = e
        self.n_channels_ = c
        self.n_samples_ = t
        self.midpoints_ = np.arange(0, t, t // (self.n_times - 1)) / self.fs
        self.n_fft_ = (self.n_freqs - 1) * 2
        self.freqs_ = np.fft.rfftfreq(self.n_fft_) * self.fs

        if self.baseline is None:  # use whole epoch
            self.baseline_ = np.arange(0, self.n_times)
        else:
            # find corresponding closest times
            tmp = [np.abs(self.midpoints_ - v).argmin() for v in self.baseline]
            self.baseline_ = np.arange(*tmp)

        stft = []
        for epoch in range(e):
            stft.append(self._stft(epochs[epoch, :, :]))
        stft = np.stack(stft, axis=0)

        ref = stft[:, self.baseline_, :, :].mean(axis=(0, 1))
        erds = stft / ref - 1

        if sig:
            lower = np.empty(erds.shape[1:])
            upper = np.empty(erds.shape[1:])
            for freq in range(len(self.freqs_)):
                for chan in range(c):
                    for time in range(len(self.midpoints_)):
                        cl, cu = bootstrap(erds[:, time, chan, freq])
                        lower[time, chan, freq], upper[time, chan, freq] = cl, cu
            self.cl_ = lower.transpose(2, 1, 0)
            self.cu_ = upper.transpose(2, 1, 0)
        self.erds_ = erds.mean(axis=0).transpose(2, 1, 0)
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

    def plot(self, channels=None, f_min=0, f_max=30, nrows=None, ncols=None,
             sig=True):
        """Plot ERDS maps.

        Parameters
        ----------
        channels : array or None
            Channels to display. If None, ERDS maps for all channels are
            displayed.
        """
        if channels is None:
            channels = range(self.n_channels_)
        f_max = min(f_max, self.fs / 2)
        c = self.n_freqs / (self.fs / 2)

        nrows = np.ceil(np.sqrt(self.n_channels_)) if nrows is None else nrows
        ncols = np.ceil(np.sqrt(self.n_channels_)) if ncols is None else ncols

        mask = np.ones(self.erds_.shape)
        if sig and self.cl_ is not None and self.cu_ is not None:
            notsig = np.logical_and(self.cl_ < 0, self.cu_ > 0)
            mask[notsig] = np.nan
        erds = self.erds_ * mask

        fig = plt.figure()
        for idx, ch in enumerate(channels):
            plt.subplot(nrows, ncols, idx + 1)
            plt.imshow(erds[f_min * c:f_max * c, ch],
                       origin="lower", aspect="auto", interpolation="none",
                       cmap=plt.get_cmap("jet_r"), vmin=-1, vmax=1.5,
                       extent=[0, self.n_samples_ / self.fs, f_min, f_max])
            plt.title(str(ch + 1), fontweight="bold")
            if idx >= self.n_channels_ - ncols:  # xlabel only in bottom row
                plt.xlabel("t (s)", fontsize=10)
            if idx % ncols == 0:  # ylabel only in left column
                plt.ylabel("f (Hz)", fontsize=10)
            plt.tick_params(labelsize=10)
        return fig
