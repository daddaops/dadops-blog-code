import numpy as np

np.random.seed(7)
# Generate linearly separable 2D data
n = 20
X_pos = np.random.randn(n // 2, 2) + np.array([1.5, 1.5])
X_neg = np.random.randn(n // 2, 2) + np.array([-1.5, -1.5])
X = np.vstack([X_pos, X_neg])
y = np.array([1]*10 + [-1]*10, dtype=float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

# Train with gradient descent for many iterations
w = np.zeros(2)
lr = 0.05
directions = []
for step in range(20000):
    logits = y * (X @ w)
    grad = -X.T @ (y * (1 - sigmoid(logits))) / n
    w = w - lr * grad
    if step % 500 == 0:
        directions.append(w / np.linalg.norm(w))

# At convergence, GD direction should match the SVM max-margin direction
from numpy.linalg import norm

gd_dir = directions[-1]
print(f"GD weight direction:  ({gd_dir[0]:.4f}, {gd_dir[1]:.4f})")
print(f"||w|| after 20k steps: {norm(w):.1f} (diverging, as expected)")
print(f"Direction change (last 5k steps): {norm(directions[-1] - directions[-2]):.6f}")
# The direction stabilizes while the norm grows — GD diverges toward max margin
