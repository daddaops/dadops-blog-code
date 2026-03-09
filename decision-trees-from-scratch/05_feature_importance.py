import numpy as np

# --- Random Forest (from block 4, needed for this script) ---
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

# --- Feature importance functions ---
def impurity_importance(forest, X, y):
    """Average Gini reduction per feature across all trees."""
    n_feat = X.shape[1]
    imp = np.zeros(n_feat)
    def traverse(node, X_sub, y_sub):
        if isinstance(node, int) or len(y_sub) == 0:
            return
        f, t = node['f'], node['t']
        mask = X_sub[:, f] <= t
        n = len(y_sub)
        pg = 1 - np.sum((np.bincount(y_sub, minlength=2) / n) ** 2)
        nl, nr = mask.sum(), (~mask).sum()
        gl = 1 - np.sum((np.bincount(y_sub[mask], minlength=2)
                         / max(nl, 1)) ** 2) if nl else 0
        gr = 1 - np.sum((np.bincount(y_sub[~mask], minlength=2)
                         / max(nr, 1)) ** 2) if nr else 0
        imp[f] += n * (pg - nl/n*gl - nr/n*gr)
        traverse(node['L'], X_sub[mask], y_sub[mask])
        traverse(node['R'], X_sub[~mask], y_sub[~mask])
    for tree in forest.trees_:
        traverse(tree, X, y)
    total = imp.sum()
    return imp / total if total > 0 else imp

def permutation_importance(forest, X, y, n_repeats=10):
    """Accuracy drop when each feature is shuffled."""
    base_acc = (forest.predict(X) == y).mean()
    result = np.zeros(X.shape[1])
    for f in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_shuf = X.copy()
            np.random.shuffle(X_shuf[:, f])
            drops.append(base_acc - (forest.predict(X_shuf) == y).mean())
        result[f] = np.mean(drops)
    return result

# 10 features: first 3 relevant, last 7 noise
np.random.seed(42)
X_t = np.random.randn(200, 10)
y_t = ((X_t[:,0] > 0) ^ (X_t[:,1] > 0) ^ (X_t[:,2] > 0)).astype(int)

rf_t = RandomForest(n_trees=50, max_depth=6).fit(X_t, y_t)
imp_i = impurity_importance(rf_t, X_t, y_t)
perm_i = permutation_importance(rf_t, X_t, y_t)

print("Feature | Impurity | Permutation | Type")
print("--------|----------|-------------|-------")
for i in range(10):
    tag = "SIGNAL" if i < 3 else "noise"
    print(f"   {i}    |  {imp_i[i]:.3f}  |    {perm_i[i]:.3f}    | {tag}")
