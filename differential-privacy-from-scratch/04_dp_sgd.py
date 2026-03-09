import numpy as np

def dp_sgd_step(X_batch, y_batch, weights, clip_norm, noise_sigma, lr):
    """One step of DP-SGD with per-example clipping and noise."""
    clipped_grads = []
    for xi, yi in zip(X_batch, y_batch):
        # Per-example logistic regression gradient
        pred = 1.0 / (1.0 + np.exp(-xi @ weights))
        grad = (pred - yi) * xi

        # Clip to bound sensitivity
        grad_norm = np.linalg.norm(grad)
        if grad_norm > clip_norm:
            grad = grad * (clip_norm / grad_norm)
        clipped_grads.append(grad)

    # Average clipped gradients + calibrated Gaussian noise
    avg_grad = np.mean(clipped_grads, axis=0)
    noise = np.random.normal(0, noise_sigma * clip_norm / len(X_batch),
                             size=weights.shape)
    return weights - lr * (avg_grad + noise)

# Synthetic 2D classification data
np.random.seed(42)
X = np.vstack([np.random.randn(50, 2) + [2, 2],
               np.random.randn(50, 2) + [-2, -2]])
y = np.array([1]*50 + [0]*50, dtype=float)

# Train both models
w_sgd = np.zeros(2)
w_dp = np.zeros(2)
for epoch in range(50):
    idx = np.random.permutation(100)
    for i in range(0, 100, 20):
        batch = idx[i:i+20]
        # Standard SGD
        preds = 1.0 / (1.0 + np.exp(-X[batch] @ w_sgd))
        grad = X[batch].T @ (preds - y[batch]) / 20
        w_sgd -= 0.1 * grad
        # DP-SGD (clip=1.0, sigma=1.0)
        w_dp = dp_sgd_step(X[batch], y[batch], w_dp,
                           clip_norm=1.0, noise_sigma=1.0, lr=0.1)

acc_sgd = np.mean((1/(1+np.exp(-X @ w_sgd)) > 0.5) == y)
acc_dp = np.mean((1/(1+np.exp(-X @ w_dp)) > 0.5) == y)
print(f"Standard SGD accuracy: {acc_sgd:.1%}")
print(f"DP-SGD accuracy:       {acc_dp:.1%}")
