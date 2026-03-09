import numpy as np

def gradient_boost_regressor(X, y, n_rounds=100, lr=0.1, max_depth=3):
    """Gradient boosting for regression (MSE loss)."""
    # Initialize with the mean prediction
    f_hat = np.full(len(y), np.mean(y))
    trees = []
    init_pred = np.mean(y)

    for _ in range(n_rounds):
        # Negative gradient of MSE loss = residuals
        residuals = y - f_hat

        # Fit a shallow tree to the residuals
        tree = build_regression_tree(X, residuals, max_depth=max_depth)
        trees.append(tree)

        # Update predictions with learning rate
        for i in range(len(X)):
            f_hat[i] += lr * predict_reg(tree, X[i])

    return init_pred, trees, lr

def build_regression_tree(X, y, depth=0, max_depth=3, min_samples=5):
    """Regression tree: leaf value = mean of targets in that region."""
    if depth >= max_depth or len(y) < min_samples:
        return {"leaf": True, "value": np.mean(y)}

    best_mse, best_feat, best_thresh = float("inf"), None, None
    n = len(y)

    for feat in range(X.shape[1]):
        thresholds = np.unique(X[:, feat])
        # Sample thresholds for speed on large datasets
        if len(thresholds) > 50:
            thresholds = np.quantile(X[:, feat], np.linspace(0, 1, 50))
        for thresh in thresholds:
            left = X[:, feat] <= thresh
            if left.sum() < min_samples or (~left).sum() < min_samples:
                continue
            mse = (np.var(y[left]) * left.sum()
                 + np.var(y[~left]) * (~left).sum()) / n
            if mse < best_mse:
                best_mse, best_feat, best_thresh = mse, feat, thresh

    if best_feat is None:
        return {"leaf": True, "value": np.mean(y)}

    left_mask = X[:, best_feat] <= best_thresh
    return {
        "leaf": False, "feature": best_feat, "threshold": best_thresh,
        "left": build_regression_tree(X[left_mask], y[left_mask],
                                      depth+1, max_depth, min_samples),
        "right": build_regression_tree(X[~left_mask], y[~left_mask],
                                       depth+1, max_depth, min_samples),
    }

def predict_reg(tree, x):
    if tree["leaf"]:
        return tree["value"]
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_reg(tree["left"], x)
    return predict_reg(tree["right"], x)

# For classification, replace residuals with pseudo-residuals:
# p = sigmoid(f_hat)  →  pseudo_residual = y - p
# This is the negative gradient of log-loss: -d/dF [-y*log(p) - (1-y)*log(1-p)]

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + np.random.randn(100) * 0.1
    init_pred, trees, lr = gradient_boost_regressor(X, y, n_rounds=50, lr=0.1)
    preds = np.full(len(y), init_pred)
    for tree in trees:
        for i in range(len(X)):
            preds[i] += lr * predict_reg(tree, X[i])
    mse = np.mean((preds - y) ** 2)
    print(f"Gradient boosting MSE: {mse:.4f}")
