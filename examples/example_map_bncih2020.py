import numpy as np
from scipy.io import loadmat
from urllib.request import urlretrieve
import os.path
import matplotlib.pyplot as plt

from erds import Erds


def load_bncih2020(fname, id):
    """Load .mat file from BNCI Horizon 2020 database.
    """
    data = loadmat(fname)
    fs = data["data"][0, 0]["fs"][0, 0]
    eeg = []
    y = []
    triggers = []
    data_len = 0

    for run in range(3, 9):
        eeg.append(data["data"][0, run]["X"][0, 0])
        y.append(data["data"][0, run]["y"][0, 0])
        triggers.append(data["data"][0, run]["trial"][0, 0] + data_len)
        data_len += eeg[-1].shape[0]

    eeg = np.concatenate(eeg).T
    y = np.concatenate(y).squeeze()
    triggers = np.concatenate(triggers).squeeze()
    fs = float(fs)
    X = cut_segments(eeg, triggers, 0, 6 * fs)
    return X[:, :22, :], y, triggers, fs


def cut_segments(x2d, tr, start, stop):
    segment = np.arange(start, stop, dtype=int)
    return np.concatenate([x2d[np.newaxis, :, t + segment] for t in tr])

def dot_special(x2d, x3d):
    return np.concatenate([x2d.T.dot(x3d[i, ...])[np.newaxis, ...]
                           for i in range(x3d.shape[0])])


fname = "A01T.mat"
if not os.path.isfile(fname):
    url = "http://bnci-horizon-2020.eu/database/data-sets/001-2014/" + fname
    fname, headers = urlretrieve(url, fname)

X, y, triggers, fs = load_bncih2020(fname, "2014-001")

labels = {1: "left", 2: "right", 3: "foot", 4: "tongue"}

# compute three Laplace derivations (C3, Cz, C4)
lap = np.zeros((3, 22))
lap[0, 7] = 1
lap[1, 9] = 1
lap[2, 11] = 1
lap[0, [1, 6, 8, 13]] = -0.25
lap[1, [3, 8, 10, 15]] = -0.25
lap[2, [5, 10, 12, 17]] = -0.25
X_lap = dot_special(lap.T, X)

for label in labels:
    maps = Erds(fs=fs, baseline=[0.5, 1.5])
    maps.fit(X_lap[y == label, :, :])
    fig = maps.plot(nrows=1, ncols=3)
    fig.suptitle(labels[label])

plt.show()
