import numpy as np
from collections import deque

def dbscan(X, eps, min_pts):
    n = len(X)
    labels = np.full(n, -1)       # -1 means noise (unassigned)
    cluster_id = 0

    # Precompute pairwise distances (naive O(n^2) approach)
    dists = np.linalg.norm(X[:, None] - X[None], axis=2)

    for i in range(n):
        if labels[i] != -1:       # Already assigned
            continue

        # Find neighbors within eps
        neighbors = np.where(dists[i] <= eps)[0]

        if len(neighbors) < min_pts:
            continue               # Not a core point — leave as noise for now

        # Start a new cluster via BFS
        labels[i] = cluster_id
        queue = deque(neighbors[neighbors != i])

        while queue:
            j = queue.popleft()
            if labels[j] == -1:    # Was noise — claim it for this cluster
                labels[j] = cluster_id
            elif labels[j] != -1 and labels[j] != cluster_id:
                continue           # Already in another cluster
            else:
                continue

            j_neighbors = np.where(dists[j] <= eps)[0]
            if len(j_neighbors) >= min_pts:   # j is also a core point
                for k in j_neighbors:
                    if labels[k] == -1:
                        queue.append(k)

        cluster_id += 1

    return labels

# Test on moons — DBSCAN nails it
from sklearn.datasets import make_moons
X, _ = make_moons(300, noise=0.06, random_state=42)
labels = dbscan(X, eps=0.15, min_pts=5)

n_clusters = len(set(labels) - {-1})
n_noise = np.sum(labels == -1)
print(f"Clusters found: {n_clusters}, Noise points: {n_noise}")
# Clusters found: 2, Noise points: 3
