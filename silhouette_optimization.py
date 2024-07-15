#!/usr/bin/env python3

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

cmap = "tab10"
norm = colors.BoundaryNorm(range(2, 10), ncolors=8)

data = load_wine()
X = data.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sil_scores = []
cluster_counts = []
eps_candidates = np.linspace(0.01, 10, 1000)

for eps in eps_candidates:
    dbscan = DBSCAN(eps=eps)
    pred = dbscan.fit_predict(X_scaled)

    cluster_count = pred.max() + 1
    cluster_counts.append(cluster_count)
    if cluster_count <= 1:
        sil_scores.append(float("nan"))
        continue

    sil_score = silhouette_score(X_scaled, pred)
    sil_scores.append(sil_score)

    # print(f"Silhouette score s={sil_score:.4f} for eps={eps:.2f}, ({pred.max() + 1} clusters)")

fig, ax = plt.subplots()
cs = ax.scatter(eps_candidates, sil_scores, marker="o", c=cluster_counts, cmap=cmap, norm=norm)
fig.colorbar(cs, label="# clusters")

ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel(r"$s$")
ax.set_title("Silhouette score optimization of DBSCAN")
# plt.show()
plt.savefig("opt_a.png", bbox_inches="tight", dpi=300)
plt.close()

sil_scores = []
n_clusters_candidates = (2, 3, 4, 5, 6, 7)

for n_clusters in n_clusters_candidates:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    pred = kmeans.fit_predict(X_scaled)
    if np.all(pred == pred[0]):
        sil_scores.append(float("nan"))
        continue

    sil_score = silhouette_score(X_scaled, pred)
    sil_scores.append(sil_score)

    # print(f"Silhouette score s={sil_score:.4f} for n_clusters={n_clusters}")

plt.scatter(n_clusters_candidates, sil_scores)
plt.xlabel("# clusters")
plt.ylabel(r"$s$")
plt.title("Silhouette score optimization of KMeans")
# plt.show()
plt.savefig("opt_b.png", bbox_inches="tight", dpi=300)
plt.close()
