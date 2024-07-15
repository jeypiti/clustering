#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

cmap = "tab10"
norm = BoundaryNorm(range(10), ncolors=10)

seed = 1
X, _ = make_blobs(
    n_samples=1000,
    centers=4,
    random_state=seed,
)

cluster_count = (2, 3, 4, 5, 6)
sil_score_by_cluster_count = []

for n_clusters in cluster_count:
    pred = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(X)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
    ax2.scatter(
        X[:, 0],
        X[:, 1],
        marker=".",
        c=pred,
        cmap=cmap,
        norm=norm,
    )
    ax2.set_xlabel("Feature A")
    ax2.set_ylabel("Feature B")

    sil_score = silhouette_score(X, pred)
    sil_samples = silhouette_samples(X, pred)
    sil_score_by_cluster_count.append(sil_score)

    y_lower = 0
    for i in range(n_clusters):
        ith_silhouette = sil_samples[pred == i]
        ith_silhouette.sort()

        y_upper = y_lower + ith_silhouette.shape[0]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_silhouette,
        )

        y_lower = y_upper + X.shape[0] / 100

    ax1.axvline(sil_score, color="black", linestyle="--")
    ax1.set_xlim((-0.1, 1))
    ax1.set_yticks([])
    ax1.set_xlabel(r"$s$")

    # fig.suptitle(f"K-Means clustering, {n_clusters} clusters")
    fig.savefig(f"sil{n_clusters}.png", bbox_inches="tigh t", dpi=300)

# plt.show()
