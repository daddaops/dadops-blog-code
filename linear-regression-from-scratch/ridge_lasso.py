import numpy as np

# Data with 3 real features, 3 irrelevant ones
np.random.seed(42)
n = 100
X = np.random.randn(n, 6)
true_w = np.array([4.0, -2.0, 3.0, 0.0, 0.0, 0.0])  # only first 3 matter
y = X @ true_w + 1.0 + np.random.randn(n) * 0.5

# Prepend ones column for bias
X_aug = np.column_stack([np.ones(n), X])

def ridge(X, y, lam):
    d = X.shape[1]
    I = np.eye(d)
    I[0, 0] = 0  # don't regularize bias
    return np.linalg.inv(X.T @ X + lam * I) @ X.T @ y

def lasso_cd(X, y, lam, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        for j in range(d):
            residual = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ residual / n
            xj_sq = np.sum(X[:, j] ** 2) / n
            if j == 0:  # don't regularize bias
                w[j] = rho / xj_sq
            else:
                w[j] = np.sign(rho) * max(abs(rho) - lam / (2 * n), 0) / xj_sq
    return w

print("True weights: [bias=1.0, 4.0, -2.0, 3.0, 0.0, 0.0, 0.0]\n")

for lam in [0.01, 0.1, 1.0, 10.0]:
    w_r = ridge(X_aug, y, lam)
    w_l = lasso_cd(X_aug, y, lam)
    print(f"λ = {lam:>5.2f}")
    print(f"  Ridge: {np.round(w_r, 2)}")
    print(f"  Lasso: {np.round(w_l, 2)}")
    lasso_zeros = np.sum(np.abs(w_l[1:]) < 0.01)
    print(f"  Lasso zeros: {lasso_zeros}/6 features\n")
