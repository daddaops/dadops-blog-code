import numpy as np
import copy

# --- DecisionTree class (from block 2, needed for this script) ---
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

# --- Pruning functions ---
def count_leaves(node):
    if 'value' in node:
        return 1
    return count_leaves(node['left']) + count_leaves(node['right'])

def subtree_impurity(node, X, y):
    """Total weighted Gini impurity across all leaves."""
    if 'value' in node:
        n = len(y)
        if n == 0:
            return 0
        return n * (1 - np.sum((np.bincount(y, minlength=2) / n) ** 2))
    mask = X[:, node['feature']] <= node['threshold']
    return (subtree_impurity(node['left'], X[mask], y[mask]) +
            subtree_impurity(node['right'], X[~mask], y[~mask]))

def effective_alpha(node, X, y):
    """Alpha at which pruning this subtree becomes optimal."""
    if 'value' in node:
        return float('inf')
    n = len(y)
    counts = np.bincount(y, minlength=2)
    r_leaf = n * (1 - np.sum((counts / max(n, 1)) ** 2))
    r_sub = subtree_impurity(node, X, y)
    n_leaves = count_leaves(node)
    return (r_leaf - r_sub) / max(n_leaves - 1, 1)

def cost_complexity_prune(node, X, y, alpha):
    """Remove subtrees where complexity cost exceeds impurity gain."""
    if 'value' in node:
        return node
    mask = X[:, node['feature']] <= node['threshold']
    node['left'] = cost_complexity_prune(
        node['left'], X[mask], y[mask], alpha)
    node['right'] = cost_complexity_prune(
        node['right'], X[~mask], y[~mask], alpha)
    if effective_alpha(node, X, y) <= alpha:
        return {'value': int(np.bincount(y).argmax())}
    return node

# --- Demo ---
np.random.seed(42)
X = np.vstack([np.random.randn(50, 2) + [1, 1],
               np.random.randn(50, 2) + [-1, -1]])
y = np.array([0]*50 + [1]*50)

# Full tree (no depth limit) vs pruned
full = DecisionTree(max_depth=20, min_samples_split=1).fit(X, y)
print(f"Full tree:  {count_leaves(full.tree_)} leaves, "
      f"train acc = {(full.predict(X) == y).mean():.1%}")

pruned = copy.deepcopy(full)
pruned.tree_ = cost_complexity_prune(pruned.tree_, X, y, alpha=0.02)
print(f"Pruned:     {count_leaves(pruned.tree_)} leaves, "
      f"train acc = {(pruned.predict(X) == y).mean():.1%}")
