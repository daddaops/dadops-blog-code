import numpy as np

def xgboost_tree(X, grads, hessians, depth=0, max_depth=4,
                  lam=1.0, gamma=0.0, min_child_weight=1.0):
    """Build one XGBoost-style tree using second-order gradients."""
    G, H = np.sum(grads), np.sum(hessians)

    # Optimal leaf weight: w* = -G / (H + lambda)
    if depth >= max_depth or len(grads) < 2 or H < min_child_weight:
        return {"leaf": True, "weight": -G / (H + lam)}

    best_gain, best_feat, best_thresh = -float("inf"), None, None

    for feat in range(X.shape[1]):
        # Sort by feature value for efficient threshold search
        order = np.argsort(X[:, feat])
        g_left, h_left = 0.0, 0.0

        for i in range(len(order) - 1):
            idx = order[i]
            g_left += grads[idx]
            h_left += hessians[idx]
            g_right = G - g_left
            h_right = H - h_left

            # Skip if child is too small
            if h_left < min_child_weight or h_right < min_child_weight:
                continue
            # Skip duplicate thresholds
            if X[order[i], feat] == X[order[i+1], feat]:
                continue

            # The XGBoost split gain formula
            gain = 0.5 * (g_left**2 / (h_left + lam)
                        + g_right**2 / (h_right + lam)
                        - G**2 / (H + lam)) - gamma

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = (X[order[i], feat] + X[order[i+1], feat]) / 2

    # No worthwhile split found (gain includes regularization penalty)
    if best_feat is None or best_gain <= 0:
        return {"leaf": True, "weight": -G / (H + lam)}

    left_mask = X[:, best_feat] <= best_thresh
    return {
        "leaf": False, "feature": best_feat, "threshold": best_thresh,
        "left": xgboost_tree(X[left_mask], grads[left_mask], hessians[left_mask],
                             depth+1, max_depth, lam, gamma, min_child_weight),
        "right": xgboost_tree(X[~left_mask], grads[~left_mask], hessians[~left_mask],
                              depth+1, max_depth, lam, gamma, min_child_weight),
    }

def predict_xgb(tree, x):
    if tree["leaf"]:
        return tree["weight"]
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_xgb(tree["left"], x)
    return predict_xgb(tree["right"], x)

# Full XGBoost training loop:
# f_hat = np.zeros(n)
# for round in range(n_rounds):
#     grads = compute_gradient(y, f_hat)    # g_i = p_i - y_i for log-loss
#     hessians = compute_hessian(y, f_hat)  # h_i = p_i * (1 - p_i)
#     tree = xgboost_tree(X, grads, hessians, lam=1.0, gamma=0.1)
#     for i in range(n):
#         f_hat[i] += lr * predict_xgb(tree, X[i])

if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    # XGBoost binary classification with log-loss
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    f_hat = np.zeros(n)
    lr = 0.3
    n_rounds = 20
    for r in range(n_rounds):
        p = sigmoid(f_hat)
        grads = p - y           # gradient of log-loss
        hessians = p * (1 - p)  # hessian of log-loss
        tree = xgboost_tree(X, grads, hessians, max_depth=3, lam=1.0, gamma=0.1)
        for i in range(n):
            f_hat[i] += lr * predict_xgb(tree, X[i])

    preds = (sigmoid(f_hat) > 0.5).astype(int)
    acc = np.mean(preds == y)
    print(f"XGBoost accuracy ({n_rounds} rounds): {acc:.2f}")
