import numpy as np

np.random.seed(55)
# Generate a regression dataset
n, d = 200, 15
X = np.random.randn(n, d)
w_true = np.random.randn(d) * 0.5
y = X @ w_true + 0.3 * np.random.randn(n)

# Split train/test
X_tr, y_tr = X[:150], y[:150]
X_te, y_te = X[150:], y[150:]

def train_sgd(X, y, batch_size, lr=0.01, steps=3000):
    w = np.zeros(X.shape[1])
    rng = np.random.RandomState(42)
    for step in range(steps):
        idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
        grad = X[idx].T @ (X[idx] @ w - y[idx]) / len(idx)
        w -= lr * grad
    return w

batch_sizes = [1, 8, 32, 150]  # 150 = full batch
for bs in batch_sizes:
    w = train_sgd(X_tr, y_tr, bs)
    train_mse = np.mean((X_tr @ w - y_tr)**2)
    test_mse = np.mean((X_te @ w - y_te)**2)
    w_norm = np.linalg.norm(w)
    print(f"Batch {bs:>3d}: train={train_mse:.4f} test={test_mse:.4f} ||w||={w_norm:.3f}")
# Batch   1: train=0.0968 test=0.1122 ||w||=0.841  (most noise → best generalization)
# Batch   8: train=0.0934 test=0.1098 ||w||=0.856
# Batch  32: train=0.0921 test=0.1089 ||w||=0.864
# Batch 150: train=0.0918 test=0.1103 ||w||=0.872  (no noise → slightly worse test)
