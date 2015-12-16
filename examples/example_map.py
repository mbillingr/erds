from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import erds


data = loadmat("test.mat")
X = data["X"]
y = data["y"].squeeze()

maps = erds.Erds(fs=512, baseline=[0.5, 2.5])
maps.fit(X[y == 0])

# plt.plot(maps.erds_[:, 0, 10])
maps.plot()
plt.show()
# test = maps.erds_[:, 0, 10]
# f = np.fft.fftfreq(512, 1/512)
# plt.plot(f[0:256], test[0:256])
# plt.show()
