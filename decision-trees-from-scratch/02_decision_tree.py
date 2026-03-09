import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples = min_samples_split

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        parent_imp = self._gini(y)
        n = len(y)
        for feat in range(X.shape[1]):
            vals = np.unique(X[:, feat])
            for i in range(len(vals) - 1):
                t = (vals[i] + vals[i + 1]) / 2
                left = X[:, feat] <= t
                if left.sum() == 0 or (~left).sum() == 0:
                    continue
                gain = parent_imp - (
                    left.sum() / n * self._gini(y[left]) +
                    (~left).sum() / n * self._gini(y[~left])
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t
        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth):
        if (len(np.unique(y)) == 1 or depth >= self.max_depth
                or len(y) < self.min_samples):
            return {'value': int(np.bincount(y).argmax())}
        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            return {'value': int(np.bincount(y).argmax())}
        left = X[:, feat] <= thresh
        return {
            'feature': feat, 'threshold': round(thresh, 4),
            'left':  self._build(X[left], y[left], depth + 1),
            'right': self._build(X[~left], y[~left], depth + 1),
        }

    def fit(self, X, y):
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _predict_one(self, node, x):
        if 'value' in node:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(node['left'], x)
        return self._predict_one(node['right'], x)

    def predict(self, X):
        return np.array([self._predict_one(self.tree_, x) for x in X])

# Train on 2D data
np.random.seed(42)
X = np.vstack([np.random.randn(50, 2) + [1, 1],
               np.random.randn(50, 2) + [-1, -1]])
y = np.array([0]*50 + [1]*50)

tree = DecisionTree(max_depth=4).fit(X, y)
print(f"Training accuracy: {(tree.predict(X) == y).mean():.1%}")
print(f"Root split: feature {tree.tree_['feature']}, "
      f"threshold {tree.tree_['threshold']}")
