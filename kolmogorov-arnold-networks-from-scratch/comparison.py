import numpy as np

def compare_kan_mlp(target_fn, n_train=500, x_range=(-2, 2)):
    """Compare a simple KAN vs MLP on function approximation."""
    np.random.seed(42)
    X = np.random.uniform(*x_range, (n_train, 2))
    y = target_fn(X[:, 0], X[:, 1])

    # Simple KAN: 2 univariate spline functions + sum
    # Approximate each input dimension with degree-5 polynomial (simulating spline)
    deg = 5
    B1 = np.column_stack([X[:, 0]**d for d in range(deg + 1)])  # (n, 6)
    B2 = np.column_stack([X[:, 1]**d for d in range(deg + 1)])
    B_kan = np.hstack([B1, B2])  # (n, 12)  -- 12 params
    c_kan = np.linalg.lstsq(B_kan, y, rcond=None)[0]
    kan_pred = B_kan @ c_kan
    kan_loss = np.mean((kan_pred - y)**2)

    # Simple MLP: 2 -> 20 (ReLU) -> 1
    W1 = np.random.randn(2, 20) * 0.5
    b1 = np.zeros(20)
    W2 = np.random.randn(20, 1) * 0.5
    b2 = np.zeros(1)
    lr = 0.001
    for step in range(2000):
        h = np.maximum(0, X @ W1 + b1)  # ReLU
        pred = h @ W2 + b2
        err = pred[:, 0] - y
        # Backprop
        dW2 = h.T @ err[:, None] / n_train
        db2 = err.mean()
        dh = err[:, None] * W2.T
        dh[X @ W1 + b1 < 0] = 0  # ReLU grad
        dW1 = X.T @ dh / n_train
        db1 = dh.mean(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    mlp_pred = np.maximum(0, X @ W1 + b1) @ W2 + b2
    mlp_loss = np.mean((mlp_pred[:, 0] - y)**2)
    mlp_params = 2*20 + 20 + 20*1 + 1  # 81 params

    # Extrapolation test: evaluate outside training range
    X_ext = np.random.uniform(2, 4, (100, 2))
    y_ext = target_fn(X_ext[:, 0], X_ext[:, 1])
    B_ext = np.hstack([np.column_stack([X_ext[:, 0]**d for d in range(deg+1)]),
                       np.column_stack([X_ext[:, 1]**d for d in range(deg+1)])])
    kan_ext_err = np.mean((B_ext @ c_kan - y_ext)**2)
    mlp_ext_pred = np.maximum(0, X_ext @ W1 + b1) @ W2 + b2
    mlp_ext_err = np.mean((mlp_ext_pred[:, 0] - y_ext)**2)

    print(f"KAN: 12 params, train loss={kan_loss:.4f}, extrap error={kan_ext_err:.4f}")
    print(f"MLP: {mlp_params} params, train loss={mlp_loss:.4f}, extrap error={mlp_ext_err:.4f}")

compare_kan_mlp(lambda x, y: np.sin(x * y) + np.cos(x))
# KAN: 12 params, train loss=0.3876, extrap error=116.3655
# MLP: 81 params, train loss=0.2445, extrap error=4.7942
