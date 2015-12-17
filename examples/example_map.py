from scipy.io import loadmat
import matplotlib.pyplot as plt
import erds


data = loadmat("test.mat")
X = data["X"]
y = data["y"].squeeze()

maps = erds.Erds(n_freqs=257, n_times=64, fs=512, baseline=[0.5, 2.5])
maps.fit(X[y == 0], sig="boot")
maps.plot()
maps.fit(X[y == 0], sig="log")
maps.plot()
# maps.fit(X[y == 0], sig="boot")
# maps.plot()
plt.show()
