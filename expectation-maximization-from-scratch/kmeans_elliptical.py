"""
K-Means on Elliptical Clusters

Shows that k-means forces spherical Voronoi boundaries, struggling
with elongated/tilted clusters. Motivates the need for EM/GMM.

Blog post: https://dadops.dev/blog/expectation-maximization-from-scratch/
"""
import numpy as np
from itertools import permutations

# Generate 3 clusters with DIFFERENT covariance structures
np.random.seed(42)
# Cluster 1: circular
c1 = np.random.randn(150, 2) * 0.8 + [0, 0]
# Cluster 2: elongated horizontally
c2 = np.random.randn(150, 2) @ [[2.5, 0], [0, 0.4]] + [5, 0]
# Cluster 3: elongated diagonally
angle = np.pi / 4
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]])
c3 = (np.random.randn(150, 2) @ [[2.0, 0], [0, 0.3]]) @ R.T + [2, 4]

X = np.vstack([c1, c2, c3])
true_labels = np.repeat([0, 1, 2], 150)

# Run k-means (Lloyd's algorithm)
centroids = X[np.random.choice(len(X), 3, replace=False)]
for _ in range(20):
    dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
    labels = np.argmin(dists, axis=1)
    centroids = np.array([X[labels == k].mean(axis=0) for k in range(3)])

# Measure accuracy (best permutation matching)
acc = max(np.mean(labels == np.array(perm)[true_labels])
          for perm in permutations(range(3)))
print(f"K-means accuracy on elliptical clusters: {acc:.1%}")
# K-means forces spherical Voronoi boundaries — misclassifying
# points in the elongated tails of non-circular clusters
