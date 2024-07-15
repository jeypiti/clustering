#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles, make_blobs

cmap = "tab10"
norm = BoundaryNorm(range(10), ncolors=10)
seed = 123

X, Y = make_circles(n_samples=1000, factor=0.6, noise=0.05, random_state=seed)

dbscan = DBSCAN(eps=0.15, min_samples=10)
pred = dbscan.fit_predict(X)

plt.scatter(
    X[:, 0],
    X[:, 1],
    marker="o",
    c=pred,
    cmap=cmap,
    norm=norm,
)
plt.xlabel("Feature A")
plt.ylabel("Feature B")
# plt.show()
plt.savefig("dbscan_a.png", bbox_inches="tight", dpi=300)
plt.close()

X, Y = make_blobs(n_samples=1000, centers=2, cluster_std=2.61, random_state=seed)

dbscan = DBSCAN(eps=2.1, min_samples=50)
pred = dbscan.fit_predict(X)

pred[pred < 0] = 2

plt.scatter(
    X[:, 0],
    X[:, 1],
    marker="o",
    c=pred,
    cmap=cmap,
    norm=norm,
)
plt.xlabel("Feature A")
plt.ylabel("Feature B")
# plt.show()
plt.savefig("dbscan_b.png", bbox_inches="tight", dpi=300)
plt.close()

X, Y = make_blobs(n_samples=1000, centers=2, cluster_std=(0.1, 3), random_state=seed)

dbscan = DBSCAN(eps=1.5, min_samples=50)
pred = dbscan.fit_predict(X)

pred[pred < 0] = 2

plt.scatter(
    X[:, 0],
    X[:, 1],
    marker="o",
    c=pred,
    cmap=cmap,
    norm=norm,
)
plt.xlabel("Feature A")
plt.ylabel("Feature B")
# plt.show()
plt.savefig("dbscan_c.png", bbox_inches="tight", dpi=300)
plt.close()
