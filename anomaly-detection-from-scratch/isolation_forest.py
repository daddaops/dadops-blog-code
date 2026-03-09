"""Isolation Forest from Scratch.

Anomaly detection via random binary trees. Anomalies are 'few and different',
so they are isolated (alone in a partition) in fewer random splits than
normal points surrounded by similar neighbors.

O(n log n) time — much faster than LOF's O(n^2) pairwise distances.
Applied to 10-dimensional data where distance-based methods struggle.
"""
import random
import math

def harmonic(n):
    """Approximate harmonic number H(n) ~ ln(n) + 0.5772."""
    return math.log(n) + 0.5772156649 if n > 1 else 0

def avg_path_bst(n):
    """Expected path length of unsuccessful search in BST of size n."""
    if n <= 1:
        return 0
    return 2 * harmonic(n - 1) - 2 * (n - 1) / n

class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.feature = None
        self.split = None
        self.left = None
        self.right = None
        self.size = 0  # external node: number of points that reached here

    def fit(self, data, depth=0):
        n = len(data)
        if n <= 1 or depth >= self.max_depth:
            self.size = n
            return self
        # Pick random feature and random split value
        n_features = len(data[0])
        self.feature = random.randint(0, n_features - 1)
        col = [row[self.feature] for row in data]
        lo, hi = min(col), max(col)
        if lo == hi:
            self.size = n
            return self
        self.split = random.uniform(lo, hi)
        left_data = [row for row in data if row[self.feature] < self.split]
        right_data = [row for row in data if row[self.feature] >= self.split]
        self.left = IsolationTree(self.max_depth).fit(left_data, depth + 1)
        self.right = IsolationTree(self.max_depth).fit(right_data, depth + 1)
        return self

    def path_length(self, point, depth=0):
        if self.feature is None:  # leaf node
            return depth + avg_path_bst(self.size)
        if point[self.feature] < self.split:
            return self.left.path_length(point, depth + 1)
        return self.right.path_length(point, depth + 1)

class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees = []

    def fit(self, data):
        max_depth = math.ceil(math.log2(self.sample_size))
        for _ in range(self.n_trees):
            sample = random.sample(data, min(self.sample_size, len(data)))
            tree = IsolationTree(max_depth).fit(sample)
            self.trees.append(tree)
        return self

    def anomaly_scores(self, data):
        c = avg_path_bst(self.sample_size)
        scores = []
        for point in data:
            avg_path = sum(t.path_length(point) for t in self.trees) / self.n_trees
            # Score in [0, 1]: close to 1 = anomaly, close to 0.5 = normal
            scores.append(2 ** (-avg_path / c) if c > 0 else 0.5)
        return scores

if __name__ == "__main__":
    # Test on 10-dimensional data (where distance methods struggle)
    random.seed(99)
    normal = [[random.gauss(0, 1) for _ in range(10)] for _ in range(500)]
    anomalies = [[random.uniform(3, 5) * random.choice([-1,1])
                   for _ in range(10)] for _ in range(10)]
    data = normal + anomalies
    labels = [0]*500 + [1]*10

    iforest = IsolationForest(n_trees=100, sample_size=256).fit(data)
    scores = iforest.anomaly_scores(data)
    top10 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    true_pos = sum(1 for i in top10 if labels[i] == 1)
    print(f"Top 10 flagged: {true_pos}/10 are true anomalies")
    print(f"Precision@10: {true_pos/10:.1%}")
