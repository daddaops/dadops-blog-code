import numpy as np

rng = np.random.RandomState(42)

# Dataset: class depends on BOTH a radial pattern (x1, x2)
# and a linear trend (x3)
n = 200
X = rng.randn(n, 3)
# Radial boundary on first two dims + linear on third
y = np.sign((X[:, 0]**2 + X[:, 1]**2 - 1.0) + 0.8 * X[:, 2])

def gram(X, kernel_fn):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = kernel_fn(X[i], X[j])
    return K

def kernel_ridge_accuracy(K, y, lam=0.1):
    alpha = np.linalg.solve(K + lam * np.eye(len(y)), y)
    preds = np.sign(K @ alpha)
    return np.mean(preds == y)

# RBF on spatial features only
K_spatial = gram(X[:, :2], lambda a, b: np.exp(-np.sum((a-b)**2) / 2.0))
# Linear on trend feature only
K_linear = gram(X[:, 2:], lambda a, b: np.dot(a, b))
# Composed: sum of both kernels
K_composed = K_spatial + K_linear

print(f"RBF (spatial only):  {kernel_ridge_accuracy(K_spatial, y):.1%}")
print(f"Linear (trend only): {kernel_ridge_accuracy(K_linear, y):.1%}")
print(f"Composed (sum):      {kernel_ridge_accuracy(K_composed, y):.1%}")

# Kernel alignment: how well does K match the ideal Gram matrix?
def kernel_alignment(K, y):
    y_outer = np.outer(y, y)  # ideal: K_ij = yi * yj
    num = np.sum(K * y_outer)
    denom = np.sqrt(np.sum(K * K) * np.sum(y_outer * y_outer))
    return num / denom

for name, K in [("Spatial", K_spatial), ("Linear", K_linear),
                ("Composed", K_composed)]:
    print(f"{name:<12} alignment: {kernel_alignment(K, y):.3f}")
