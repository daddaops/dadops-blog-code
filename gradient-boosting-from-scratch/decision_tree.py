import numpy as np

def gini_impurity(y):
    """Gini = 1 - sum(p_k^2) for each class k."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

def best_split(X, y):
    """Find the feature and threshold that maximize information gain."""
    best_gain, best_feat, best_thresh = -1, None, None
    parent_impurity = gini_impurity(y)
    n = len(y)

    for feat in range(X.shape[1]):
        thresholds = np.unique(X[:, feat])
        for thresh in thresholds:
            left_mask = X[:, feat] <= thresh
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            # Weighted average impurity of children
            w_left = left_mask.sum() / n
            child_impurity = (w_left * gini_impurity(y[left_mask])
                            + (1 - w_left) * gini_impurity(y[right_mask]))
            gain = parent_impurity - child_impurity

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh

    return best_feat, best_thresh, best_gain

def build_tree(X, y, depth=0, max_depth=5, min_samples=2):
    """Recursively build a decision tree."""
    # Leaf node: return majority class
    if depth >= max_depth or len(y) < min_samples or len(np.unique(y)) == 1:
        classes, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "class": classes[np.argmax(counts)]}

    feat, thresh, gain = best_split(X, y)
    if feat is None or gain <= 0:
        classes, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "class": classes[np.argmax(counts)]}

    left_mask = X[:, feat] <= thresh
    return {
        "leaf": False, "feature": feat, "threshold": thresh,
        "left": build_tree(X[left_mask], y[left_mask], depth+1, max_depth, min_samples),
        "right": build_tree(X[~left_mask], y[~left_mask], depth+1, max_depth, min_samples),
    }

def predict_one(tree, x):
    if tree["leaf"]:
        return tree["class"]
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_one(tree["left"], x)
    return predict_one(tree["right"], x)

if __name__ == "__main__":
    # Usage: tree = build_tree(X_train, y_train, max_depth=4)
    # predictions = [predict_one(tree, x) for x in X_test]
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    tree = build_tree(X, y, max_depth=4)
    preds = np.array([predict_one(tree, x) for x in X])
    acc = np.mean(preds == y)
    print(f"Decision tree accuracy: {acc:.2f}")
