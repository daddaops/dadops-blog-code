import numpy as np
from scipy.spatial import KDTree
from collections import deque
import time

def dbscan_kdtree(X, eps, min_pts):
    """DBSCAN using KD-tree for O(n log n) range queries."""
    tree = KDTree(X)
    n = len(X)
    labels = np.full(n, -1)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        neighbors = tree.query_ball_point(X[i], eps)
        if len(neighbors) < min_pts:
            continue

        labels[i] = cluster_id
        queue = deque([j for j in neighbors if j != i])

        while queue:
            j = queue.popleft()
            if labels[j] == -1:
                labels[j] = cluster_id
            else:
                continue
            j_neighbors = tree.query_ball_point(X[j], eps)
            if len(j_neighbors) >= min_pts:
                for k in j_neighbors:
                    if labels[k] == -1:
                        queue.append(k)
        cluster_id += 1
    return labels

# Benchmark naive vs KD-tree
from sklearn.datasets import make_moons
for n in [500, 2000, 5000]:
    X, _ = make_moons(n, noise=0.06, random_state=42)

    t0 = time.time()
    dists = np.linalg.norm(X[:, None] - X[None], axis=2)  # naive distance matrix
    t_naive = time.time() - t0

    t0 = time.time()
    dbscan_kdtree(X, eps=0.15, min_pts=5)
    t_tree = time.time() - t0

    print(f"n={n:>5d} | Naive dist matrix: {t_naive:.3f}s | KD-tree DBSCAN: {t_tree:.3f}s")
# n=  500 | Naive dist matrix: 0.025s | KD-tree DBSCAN: 0.021s
# n= 2000 | Naive dist matrix: 0.372s | KD-tree DBSCAN: 0.146s
# n= 5000 | Naive dist matrix: 2.520s | KD-tree DBSCAN: 0.632s
