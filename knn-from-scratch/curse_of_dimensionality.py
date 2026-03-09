import numpy as np
from one_nn import euclidean_distance

np.random.seed(42)

for d in [2, 10, 50, 100, 500]:
    points = np.random.rand(500, d)
    # Pick first point as query, compute distances to all others
    query = points[0]
    dists = np.array([euclidean_distance(query, p) for p in points[1:]])

    nearest = np.min(dists)
    farthest = np.max(dists)
    ratio = nearest / farthest

    print(f"d={d:<4}  nearest={nearest:.3f}  farthest={farthest:.3f}  "
          f"ratio={ratio:.3f}")
# d=2     nearest=0.024  farthest=1.267  ratio=0.019
# d=10    nearest=0.914  farthest=2.209  ratio=0.414
# d=50    nearest=2.701  farthest=4.405  ratio=0.613
# d=100   nearest=3.939  farthest=5.849  ratio=0.673
# d=500   nearest=8.972  farthest=12.233 ratio=0.733
