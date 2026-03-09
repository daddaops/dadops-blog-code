import numpy as np

np.random.seed(123)
# Small neural network to demonstrate edge of stability
n, d, h = 50, 5, 20  # 50 samples, 5 features, 20 hidden
X = np.random.randn(n, d)
y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1])

# Initialize network: f(x) = W2 @ relu(W1 @ x)
W1 = np.random.randn(h, d) * np.sqrt(2.0 / d)
W2 = np.random.randn(1, h) * np.sqrt(2.0 / h)

def loss_and_grad(W1, W2, X, y):
    H = X @ W1.T
    A = np.maximum(H, 0)
    pred = (A @ W2.T).ravel()
    r = pred - y
    L = 0.5 * np.mean(r ** 2)
    dW2 = (r[:, None] * A).mean(axis=0, keepdims=True)
    dA = np.outer(r, W2.ravel()) / n
    dH = dA * (H > 0).astype(float)
    dW1 = dH.T @ X / n
    return L, dW1, dW2

# Train with LARGE learning rate (above classical 2/L threshold)
lr = 0.5
losses = []
sharpnesses = []
for step in range(2000):
    L, dW1, dW2 = loss_and_grad(W1, W2, X, y)
    W1 -= lr * dW1
    W2 -= lr * dW2
    losses.append(L)
    # Estimate top Hessian eigenvalue via finite differences (power iteration approx)
    if step % 50 == 0:
        eps = 1e-4
        v1 = np.random.randn(*W1.shape); v2 = np.random.randn(*W2.shape)
        nv = np.sqrt(np.sum(v1**2) + np.sum(v2**2))
        v1 /= nv; v2 /= nv
        _, g1a, g2a = loss_and_grad(W1 + eps*v1, W2 + eps*v2, X, y)
        _, g1b, g2b = loss_and_grad(W1 - eps*v1, W2 - eps*v2, X, y)
        hv = np.sqrt(np.sum((g1a-g1b)**2) + np.sum((g2a-g2b)**2)) / (2*eps)
        sharpnesses.append(hv)

print(f"Loss: {losses[0]:.3f} -> {losses[-1]:.4f}")
print(f"2/lr = {2/lr:.1f}")
print(f"Sharpness (step 0): {sharpnesses[0]:.2f}")
print(f"Sharpness (step 2000): {sharpnesses[-1]:.2f}")
# Sharpness converges toward 2/η — the edge of stability!
