import numpy as np
from decision_tree import gini_impurity, predict_one

def random_forest(X, y, n_trees=100, max_depth=8, max_features="sqrt"):
    """Build a random forest via bagging + feature subsampling."""
    n_samples, n_features = X.shape
    n_sub = int(np.sqrt(n_features)) if max_features == "sqrt" else n_features
    trees = []

    for _ in range(n_trees):
        # Bootstrap sample (draw n_samples with replacement)
        idx = np.random.randint(0, n_samples, size=n_samples)
        X_boot, y_boot = X[idx], y[idx]

        # Build tree with feature subsampling at each split
        tree = build_tree_rf(X_boot, y_boot, max_depth=max_depth,
                             n_sub_features=n_sub)
        trees.append(tree)

    return trees

def build_tree_rf(X, y, depth=0, max_depth=8, min_samples=2, n_sub_features=None):
    """Decision tree with random feature subsampling at each split."""
    if depth >= max_depth or len(y) < min_samples or len(np.unique(y)) == 1:
        classes, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "class": classes[np.argmax(counts)]}

    # Randomly select a subset of features to consider
    all_feats = np.arange(X.shape[1])
    chosen = np.random.choice(all_feats, size=n_sub_features, replace=False)

    best_gain, best_feat, best_thresh = -1, None, None
    parent_imp = gini_impurity(y)
    n = len(y)

    for feat in chosen:
        for thresh in np.unique(X[:, feat]):
            left = X[:, feat] <= thresh
            if left.sum() == 0 or (~left).sum() == 0:
                continue
            w = left.sum() / n
            imp = w * gini_impurity(y[left]) + (1-w) * gini_impurity(y[~left])
            gain = parent_imp - imp
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh

    if best_feat is None:
        classes, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "class": classes[np.argmax(counts)]}

    left_mask = X[:, best_feat] <= best_thresh
    return {
        "leaf": False, "feature": best_feat, "threshold": best_thresh,
        "left": build_tree_rf(X[left_mask], y[left_mask], depth+1,
                              max_depth, min_samples, n_sub_features),
        "right": build_tree_rf(X[~left_mask], y[~left_mask], depth+1,
                               max_depth, min_samples, n_sub_features),
    }

def forest_predict(trees, X):
    """Majority vote across all trees."""
    all_preds = np.array([[predict_one(t, x) for x in X] for t in trees])
    # Most common prediction for each sample
    return np.array([np.bincount(all_preds[:, i].astype(int)).argmax()
                     for i in range(X.shape[0])])

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    trees = random_forest(X, y, n_trees=20, max_depth=4)
    preds = forest_predict(trees, X)
    acc = np.mean(preds == y)
    print(f"Random forest accuracy (20 trees): {acc:.2f}")
