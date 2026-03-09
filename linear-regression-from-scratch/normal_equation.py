import numpy as np

# Generate multi-feature data with known weights
np.random.seed(42)
n, d = 200, 5
true_w = np.array([3.0, -1.5, 0.0, 2.0, 0.0])  # 3 active, 2 zero
true_b = 4.0

X = np.random.randn(n, d)
y = X @ true_w + true_b + np.random.randn(n) * 0.5

# Prepend column of ones for bias
X_aug = np.column_stack([np.ones(n), X])

# Method 1: Normal equation
w_normal = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y

# Method 2: Pseudoinverse (more stable)
w_pinv = np.linalg.pinv(X_aug) @ y

print("True weights:    ", np.round([true_b] + list(true_w), 3))
print("Normal equation: ", np.round(w_normal, 3))
print("Pseudoinverse:   ", np.round(w_pinv, 3))

# Now break it: add a collinear feature (x6 = 2 * x1)
X_bad = np.column_stack([X, 2 * X[:, 0]])
X_bad_aug = np.column_stack([np.ones(n), X_bad])

cond_number = np.linalg.cond(X_bad_aug.T @ X_bad_aug)
print(f"\nCondition number with collinear feature: {cond_number:.2e}")
print("(Anything above ~1e12 means numerical trouble)")

# Pseudoinverse still works
w_pinv_bad = np.linalg.pinv(X_bad_aug) @ y
print("Pseudoinverse still finds a solution: ✓")
