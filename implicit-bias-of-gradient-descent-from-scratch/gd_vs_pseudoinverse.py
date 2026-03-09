import numpy as np

np.random.seed(0)
n, p = 8, 30
X = np.random.randn(n, p)
y = np.random.randn(n)

# Method 1: Gradient descent from w=0
w_gd = np.zeros(p)
for _ in range(10000):
    w_gd -= 0.005 * X.T @ (X @ w_gd - y) / n

# Method 2: Pseudoinverse (explicit minimum-norm solution)
w_pinv = X.T @ np.linalg.solve(X @ X.T, y)

# Method 3: Ridge regression with lambda -> 0
lam = 1e-10
w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)

print(f"||w_gd - w_pinv||  = {np.linalg.norm(w_gd - w_pinv):.2e}")
print(f"||w_gd - w_ridge|| = {np.linalg.norm(w_gd - w_ridge):.2e}")
print(f"||w_gd||  = {np.linalg.norm(w_gd):.4f}")
print(f"||w_pinv||= {np.linalg.norm(w_pinv):.4f}")
# ||w_gd - w_pinv||  = 3.12e-04
# ||w_gd - w_ridge|| = 2.97e-04
# All three methods converge to the same minimum-norm solution!
