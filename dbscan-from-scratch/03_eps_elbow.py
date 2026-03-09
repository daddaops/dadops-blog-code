import numpy as np
from scipy.spatial import KDTree

def find_eps_elbow(X, k=5):
    """Use k-distance graph to find a good eps value."""
    tree = KDTree(X)
    # Query k+1 neighbors (includes the point itself)
    dists, _ = tree.query(X, k=k + 1)
    k_dists = dists[:, -1]              # Distance to k-th neighbor
    k_dists_sorted = np.sort(k_dists)

    # Find the elbow: point of maximum curvature
    # Approximate by looking for the biggest acceleration in the curve
    diffs = np.diff(k_dists_sorted)
    accel = np.diff(diffs)
    elbow_idx = np.argmax(accel) + 2
    eps_estimate = k_dists_sorted[elbow_idx]

    print(f"k-distance elbow at index {elbow_idx}/{len(X)}")
    print(f"Suggested eps: {eps_estimate:.4f}")
    lo = max(0, elbow_idx - 5)
    hi = min(len(k_dists_sorted) - 1, elbow_idx + 5)
    print(f"Range around elbow: [{k_dists_sorted[lo]:.4f}, {k_dists_sorted[hi]:.4f}]")
    return eps_estimate

from sklearn.datasets import make_moons
X, _ = make_moons(300, noise=0.06, random_state=42)
eps = find_eps_elbow(X, k=5)
# k-distance elbow at index 283/300
# Suggested eps: 0.1412
# Range around elbow: [0.1178, 0.1830]
