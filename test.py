import matplotlib.pyplot as plt
import numpy as np


def create_spiral_data(n_points, n_classes, noise=0.5, distance=100.0):
    X = np.zeros((n_points * n_classes, 2))
    y = np.zeros(n_points * n_classes, dtype="uint8")
    for j in range(n_classes):
        ix = range(n_points * j, n_points * (j + 1))
        r = np.linspace(0.0, 1, n_points)  # radius
        t = (
            np.linspace(j * 4, (j + 1) * 4, n_points)
            + np.random.randn(n_points) * noise
        )  # theta
        X[ix] = np.c_[r * np.sin(t) * distance, r * np.cos(t) * distance]
        y[ix] = j
    return X, y


X, y = create_spiral_data(10000, 3)

plt.scatter(X[:, 0], X[:, 1], s=0.5, c=y, cmap=plt.cm.Spectral)
plt.axis("square")
plt.show()
