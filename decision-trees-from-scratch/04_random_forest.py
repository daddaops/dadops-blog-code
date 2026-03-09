import numpy as np

def build_tree(X, y, depth, max_depth, max_features=None):
    """Recursive CART with optional random feature subsets."""
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2:
        return int(np.bincount(y).argmax())
    m = X.shape[1]
    feats = np.random.choice(m, max_features or m, replace=False)
    best_gain, best_f, best_t = -1, None, None
    parent_g = 1 - np.sum((np.bincount(y, minlength=2) / len(y)) ** 2)
    for f in feats:
        vals = np.unique(X[:, f])
        for i in range(len(vals) - 1):
            t = (vals[i] + vals[i+1]) / 2
            left = X[:, f] <= t
            nl, nr = left.sum(), (~left).sum()
            if nl == 0 or nr == 0:
                continue
            gl = 1 - np.sum((np.bincount(y[left], minlength=2) / nl) ** 2)
            gr = 1 - np.sum((np.bincount(y[~left], minlength=2) / nr) ** 2)
            gain = parent_g - (nl * gl + nr * gr) / len(y)
            if gain > best_gain:
                best_gain, best_f, best_t = gain, f, t
    if best_gain <= 0:
        return int(np.bincount(y).argmax())
    mask = X[:, best_f] <= best_t
    return {
        'f': best_f, 't': best_t,
        'L': build_tree(X[mask], y[mask], depth+1, max_depth, max_features),
        'R': build_tree(X[~mask], y[~mask], depth+1, max_depth, max_features),
    }

def walk(node, x):
    if isinstance(node, int):
        return node
    return walk(node['L'] if x[node['f']] <= node['t'] else node['R'], x)

class RandomForest:
    def __init__(self, n_trees=100, max_depth=5):
        self.n_trees, self.max_depth = n_trees, max_depth

    def fit(self, X, y):
        n, m = X.shape
        mf = int(np.sqrt(m))
        self.trees_, self.oob_ = [], []
        for _ in range(self.n_trees):
            bag = np.random.choice(n, n, replace=True)
            oob = list(set(range(n)) - set(bag))
            self.trees_.append(
                build_tree(X[bag], y[bag], 0, self.max_depth, mf))
            self.oob_.append(oob)
        return self

    def predict(self, X):
        preds = np.array([[walk(t, x) for x in X] for t in self.trees_])
        return np.array([np.bincount(preds[:,i]).argmax()
                         for i in range(len(X))])

    def oob_score(self, X, y):
        """Each sample scored only by trees that didn't train on it."""
        votes = {i: [] for i in range(len(y))}
        for tree, oob in zip(self.trees_, self.oob_):
            for idx in oob:
                votes[idx].append(walk(tree, X[idx]))
        scored = {k: v for k, v in votes.items() if v}
        return np.mean([np.bincount(v).argmax() == y[k]
                        for k, v in scored.items()])

# Single tree vs Random Forest
np.random.seed(42)
X = np.vstack([np.random.randn(100, 2) + [1, 1],
               np.random.randn(100, 2) + [-1, -1]])
y = np.array([0]*100 + [1]*100)

single = build_tree(X, y, 0, max_depth=10)
single_acc = np.mean([walk(single, x) == t for x, t in zip(X, y)])
rf = RandomForest(n_trees=100, max_depth=5).fit(X, y)

print(f"Single tree: {single_acc:.1%}")       # ~100%
print(f"Forest:      {(rf.predict(X) == y).mean():.1%}")  # ~100%
print(f"OOB score:   {rf.oob_score(X, y):.1%}")           # ~96%
