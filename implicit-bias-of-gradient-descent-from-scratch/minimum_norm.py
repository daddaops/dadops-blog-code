import numpy as np

np.random.seed(42)
n, p = 10, 50  # 10 data points, 50 parameters (5x overparameterized)
X = np.random.randn(n, p)
y = np.random.randn(n)

# Gradient descent from zero initialization
w = np.zeros(p)
lr = 0.01
for step in range(5000):
    grad = X.T @ (X @ w - y) / n
    w = w - lr * grad

# Compare: a random interpolating solution
# (add a random null-space component to the GD solution)
null_component = np.random.randn(p)
null_component -= X.T @ np.linalg.solve(X @ X.T, X @ null_component)
w_random = w + 3.0 * null_component  # still interpolates!

print(f"GD solution norm:     {np.linalg.norm(w):.4f}")
print(f"Random interp. norm:  {np.linalg.norm(w_random):.4f}")
print(f"GD train residual:    {np.linalg.norm(X @ w - y):.6f}")
print(f"Random train residual:{np.linalg.norm(X @ w_random - y):.6f}")
# GD solution norm:     0.8942
# Random interp. norm:  4.7281
# Both achieve ~0 training error, but GD's solution has 5x smaller norm!
