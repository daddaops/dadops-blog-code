import numpy as np

rng = np.random.RandomState(42)

# --- Kernel Ridge Regression on nonlinear data ---
X_reg = np.sort(rng.uniform(-3, 3, 60))
y_reg = np.sin(X_reg) + 0.2 * rng.randn(60)

def rbf(x, y, sigma=0.5):
    return np.exp(-((x - y) ** 2) / (2 * sigma ** 2))

K = np.array([[rbf(xi, xj) for xj in X_reg] for xi in X_reg])
alpha = np.linalg.solve(K + 0.01 * np.eye(60), y_reg)

# Predict on a fine grid
X_test = np.linspace(-3, 3, 200)
y_pred = np.array([sum(a * rbf(xt, xi) for a, xi in zip(alpha, X_reg))
                    for xt in X_test])
mse_kernel = np.mean((np.sin(X_test) - y_pred) ** 2)

# Compare: linear ridge
w_lin = np.linalg.solve(X_reg[:, None].T @ X_reg[:, None]
        + 0.01 * np.eye(1), X_reg[:, None].T @ y_reg)
y_lin = X_test * w_lin[0]
mse_linear = np.mean((np.sin(X_test) - y_lin) ** 2)

print(f"Kernel ridge MSE:  {mse_kernel:.4f}")
print(f"Linear ridge MSE:  {mse_linear:.4f}")

# --- Kernel PCA on concentric circles ---
n = 200
theta = rng.uniform(0, 2 * np.pi, n)
r_inner, r_outer = 1.0, 3.0
X_circles = np.vstack([
    np.column_stack([r_inner * np.cos(theta[:100]), r_inner * np.sin(theta[:100])]),
    np.column_stack([r_outer * np.cos(theta[100:]), r_outer * np.sin(theta[100:])])
])
X_circles += 0.2 * rng.randn(n, 2)
labels = np.array([0]*100 + [1]*100)

# Gram matrix
K_c = np.array([[np.exp(-np.sum((X_circles[i]-X_circles[j])**2)/2.0)
                 for j in range(n)] for i in range(n)])
# Center: K_tilde = HKH
H = np.eye(n) - np.ones((n, n)) / n
K_tilde = H @ K_c @ H
eigvals, eigvecs = np.linalg.eigh(K_tilde)
# Take top 2 components (largest eigenvalues)
idx = np.argsort(eigvals)[::-1][:2]
Z = eigvecs[:, idx] * np.sqrt(np.abs(eigvals[idx]))

# Check if first kernel PC separates the classes
threshold = np.median(Z[:, 0])
kpca_preds = (Z[:, 0] > threshold).astype(int)
kpca_acc = max(np.mean(kpca_preds == labels), np.mean(kpca_preds != labels))
print(f"\nKernel PCA: 1st component separates circles with {kpca_acc:.1%} accuracy")
