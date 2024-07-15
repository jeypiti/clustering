#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs, make_circles

cmap = "tab10"
norm = BoundaryNorm(range(10), ncolors=10)

seed = 123
X, Y = make_blobs(n_samples=1000, centers=3, cluster_std=2, random_state=seed)

meanshift = MeanShift(bandwidth=4)
meanshift.fit(X)
pred = meanshift.predict(X)

plt.scatter(
    X[:, 0],
    X[:, 1],
    marker=".",
    c=pred,
    cmap=cmap,
    norm=norm,
)
plt.scatter(
    meanshift.cluster_centers_[:, 0],
    meanshift.cluster_centers_[:, 1],
    marker="x",
    color="black",
)
plt.xlabel("Feature A")
plt.ylabel("Feature B")
plt.title("Mean shift, $b=4$")
# plt.show()
plt.savefig("meanshift_a.png", bbox_inches="tight", dpi=300)
plt.close()

X, Y = make_circles(n_samples=1000, factor=0.6, noise=0.05, random_state=seed)

meanshift = MeanShift(bandwidth=0.85)
meanshift.fit(X)
pred = meanshift.predict(X)

plt.scatter(
    X[:, 0],
    X[:, 1],
    marker=".",
    c=pred,
    cmap=cmap,
    norm=norm,
)
plt.scatter(
    meanshift.cluster_centers_[:, 0],
    meanshift.cluster_centers_[:, 1],
    marker="x",
    color="black",
)
plt.xlabel("Feature A")
plt.ylabel("Feature B")
plt.title("Mean shift, $b=0.85$")
# plt.show()
plt.savefig("meanshift_b.png", bbox_inches="tight", dpi=300)
