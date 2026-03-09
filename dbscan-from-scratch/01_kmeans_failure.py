import numpy as np
from sklearn.datasets import make_moons, make_circles

# Generate non-convex datasets
moons_X, moons_y = make_moons(n_samples=300, noise=0.06, random_state=42)
circles_X, circles_y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

def kmeans(X, k, seed=42):
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    for _ in range(50):
        dists = np.linalg.norm(X[:, None] - centers[None], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centers, new_centers): break
        centers = new_centers
    return labels

moon_labels = kmeans(moons_X, 2)
circle_labels = kmeans(circles_X, 2)

# K-Means splits both datasets along a straight line
# Moons: top and bottom halves instead of the two crescents
# Circles: left and right halves instead of inner and outer rings
print(f"Moons — K-Means accuracy: {max(np.mean(moon_labels == moons_y), np.mean(moon_labels != moons_y)):.0%}")
print(f"Circles — K-Means accuracy: {max(np.mean(circle_labels == circles_y), np.mean(circle_labels != circles_y)):.0%}")
# Moons — K-Means accuracy: 75%
# Circles — K-Means accuracy: 51%  (no better than random!)
