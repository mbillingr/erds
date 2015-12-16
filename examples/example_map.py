from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import erds


data = loadmat("test.mat")
X = data["X"]
y = data["y"].squeeze()

maps = erds.Erds(n_freqs=129, n_times=32, fs=512, baseline=[0.5, 2.5])
maps.fit(X[y == 0], sig=True)
maps.plot()
plt.show()
