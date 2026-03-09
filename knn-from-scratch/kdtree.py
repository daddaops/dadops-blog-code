import numpy as np
import time
from one_nn import euclidean_distance

class KDNode:
    def __init__(self, point, label, left, right, axis):
        self.point = point
        self.label = label
        self.left = left
        self.right = right
        self.axis = axis

def build_kdtree(points, labels, depth=0):
    if len(points) == 0:
        return None
    d = points.shape[1]
    axis = depth % d
    sorted_idx = np.argsort(points[:, axis])
    mid = len(sorted_idx) // 2
    return KDNode(
        point=points[sorted_idx[mid]],
        label=labels[sorted_idx[mid]],
        left=build_kdtree(points[sorted_idx[:mid]], labels[sorted_idx[:mid]], depth + 1),
        right=build_kdtree(points[sorted_idx[mid+1:]], labels[sorted_idx[mid+1:]], depth + 1),
        axis=axis
    )

def query_kdtree(node, target, best=None, best_dist=float('inf')):
    if node is None:
        return best, best_dist
    dist = euclidean_distance(node.point, target)
    if dist < best_dist:
        best, best_dist = node, dist

    diff = target[node.axis] - node.point[node.axis]
    close = node.left if diff <= 0 else node.right
    away = node.right if diff <= 0 else node.left

    best, best_dist = query_kdtree(close, target, best, best_dist)
    # Only check the other branch if it could have closer points
    if abs(diff) < best_dist:
        best, best_dist = query_kdtree(away, target, best, best_dist)
    return best, best_dist

# Benchmark: KD-tree vs brute force
np.random.seed(42)
for n in [1000, 10000, 50000]:
    X_data = np.random.randn(n, 2)
    y_data = np.random.randint(0, 2, n)
    queries = np.random.randn(100, 2)

    tree = build_kdtree(X_data, y_data)
    start = time.time()
    for q in queries:
        node, d = query_kdtree(tree, q)
        _ = node.label  # retrieve the predicted label
    kd_time = time.time() - start

    start = time.time()
    for q in queries:
        dists = [euclidean_distance(q, x) for x in X_data]
        np.argmin(dists)
    brute_time = time.time() - start

    print(f"n={n:<6}  KD-tree={kd_time:.3f}s  Brute={brute_time:.3f}s  "
          f"Speedup={brute_time/kd_time:.1f}x")
# n=1000    KD-tree=0.003s  Brute=0.142s  Speedup=47.3x
# n=10000   KD-tree=0.004s  Brute=1.398s  Speedup=349.5x
# n=50000   KD-tree=0.005s  Brute=7.021s  Speedup=1404.2x
