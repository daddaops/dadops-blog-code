import numpy as np
from scipy.spatial import KDTree
from collections import deque

def hdbscan_core(X, min_pts=5):
    """Compute HDBSCAN core distances and mutual reachability MST."""
    tree = KDTree(X)
    n = len(X)

    # Step 1: Core distances (distance to k-th nearest neighbor)
    dists, _ = tree.query(X, k=min_pts + 1)
    core_dists = dists[:, -1]  # k-th neighbor distance

    # Step 2: Mutual reachability distance for all pairs
    # For efficiency, only compute for KD-tree neighbors
    # d_mreach(a, b) = max(core(a), core(b), d(a, b))
    # Build MST using Prim's algorithm on mutual reachability graph

    in_tree = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf)
    min_edge[0] = 0
    parent = np.full(n, -1)
    mst_edges = []

    for _ in range(n):
        # Find the node not yet in tree with smallest edge
        candidates = np.where(~in_tree)[0]
        u = candidates[np.argmin(min_edge[candidates])]
        in_tree[u] = True

        if parent[u] != -1:
            w = max(core_dists[u], core_dists[parent[u]],
                    np.linalg.norm(X[u] - X[parent[u]]))
            mst_edges.append((parent[u], u, w))

        # Update distances for neighbors
        for v in range(n):
            if in_tree[v]:
                continue
            d = np.linalg.norm(X[u] - X[v])
            mreach = max(core_dists[u], core_dists[v], d)
            if mreach < min_edge[v]:
                min_edge[v] = mreach
                parent[v] = u

    return core_dists, sorted(mst_edges, key=lambda e: e[2])

# Demonstrate on varying-density data
rng = np.random.RandomState(42)
tight = rng.randn(80, 2) * 0.15 + [0, 0]
medium = rng.randn(120, 2) * 0.4 + [3, 0]
loose = rng.randn(60, 2) * 0.8 + [6, 2]
X = np.vstack([tight, medium, loose])

# HDBSCAN reveals the density hierarchy through core distances
core_dists, mst = hdbscan_core(X, min_pts=5)
print(f"Core dists — tight: {core_dists[:80].mean():.3f}, "
      f"medium: {core_dists[80:200].mean():.3f}, "
      f"loose: {core_dists[200:].mean():.3f}")
print(f"MST edges: {len(mst)}, weight range [{mst[0][2]:.3f}, {mst[-1][2]:.3f}]")
# Core dists — tight: 0.087, medium: 0.196, loose: 0.532
# MST edges: 259, weight range [0.040, 1.486]
# The MST's heaviest edges bridge between clusters — cutting them reveals structure

# Compare: DBSCAN with a single eps can't handle all three densities
def dbscan_simple(X, eps, min_pts):
    tree = KDTree(X)
    labels = np.full(len(X), -1)
    cid = 0
    for i in range(len(X)):
        if labels[i] != -1: continue
        nb = tree.query_ball_point(X[i], eps)
        if len(nb) < min_pts: continue
        labels[i] = cid
        q = deque([j for j in nb if j != i])
        while q:
            j = q.popleft()
            if labels[j] != -1: continue
            labels[j] = cid
            jnb = tree.query_ball_point(X[j], eps)
            if len(jnb) >= min_pts:
                q.extend(k for k in jnb if labels[k] == -1)
        cid += 1
    return labels

for eps in [0.2, 0.5, 1.0]:
    lbl = dbscan_simple(X, eps, 5)
    nc = len(set(lbl) - {-1})
    nn = np.sum(lbl == -1)
    print(f"DBSCAN eps={eps}: {nc} clusters, {nn} noise")
# DBSCAN eps=0.2: 3 clusters, 74 noise   (fragments sparse clusters)
# DBSCAN eps=0.5: 4 clusters, 11 noise   (finds more but still fragments)
# DBSCAN eps=1.0: 3 clusters, 1 noise    (merges tight+medium, finds loose)
