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
# d=2     nearest=0.024  farthest=1.040  ratio=0.023
# d=10    nearest=0.567  farthest=1.721  ratio=0.329
# d=50    nearest=2.229  farthest=3.558  ratio=0.626
# d=100   nearest=3.309  farthest=4.738  ratio=0.698
# d=500   nearest=8.635  farthest=9.833  ratio=0.878
