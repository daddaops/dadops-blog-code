import numpy as np

rng = np.random.RandomState(42)
X = rng.randn(50, 2)

def gram_matrix(X, kernel_fn):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_fn(X[i], X[j])
    return K

# Valid kernels — all eigenvalues >= 0
def k_linear(x, y): return np.dot(x, y)
def k_poly(x, y): return (np.dot(x, y) + 1) ** 3
def k_rbf(x, y): return np.exp(-np.sum((x - y)**2) / 2.0)

# Invalid "kernel" — not PSD for all point sets
def k_bad(x, y): return np.sin(np.dot(x, y))

for name, kfn in [("Linear", k_linear), ("Poly(3)", k_poly),
                   ("RBF", k_rbf), ("sin(xTy)", k_bad)]:
    K = gram_matrix(X, kfn)
    eigvals = np.linalg.eigvalsh(K)
    min_eig = eigvals.min()
    neg_count = np.sum(eigvals < -1e-10)
    status = "VALID (PSD)" if neg_count == 0 else f"INVALID ({neg_count} negative eigenvalues)"
    print(f"{name:<10} min eigenvalue: {min_eig:+.4f}  {status}")
