"""K-FAC for a 2-layer MLP."""
import numpy as np

np.random.seed(42)
# Synthetic binary classification: 100 points, 5 features
X = np.random.randn(100, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(float)

# 2-layer MLP: 5 -> 8 -> 1
W1 = np.random.randn(5, 8) * 0.3
b1 = np.zeros(8)
W2 = np.random.randn(8, 1) * 0.3
b2 = np.zeros(1)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def forward_and_loss(X, y, W1, b1, W2, b2):
    h = np.tanh(X @ W1 + b1)           # hidden activations
    logits = (h @ W2 + b2).ravel()
    probs = sigmoid(logits)
    loss = -np.mean(y * np.log(probs + 1e-8) + (1-y) * np.log(1-probs + 1e-8))
    return h, probs, loss

# K-FAC update for one step
damping = 0.01
for step in range(60):
    h, probs, loss = forward_and_loss(X, y, W1, b1, W2, b2)
    if step % 20 == 0:
        print(f"Step {step}: loss = {loss:.4f}")

    # Backprop: output gradient
    dl_dlogits = (probs - y) / len(y)  # (100,)
    dl_dW2 = h.T @ dl_dlogits.reshape(-1, 1)
    dl_db2 = dl_dlogits.sum(keepdims=True)

    dl_dh = dl_dlogits.reshape(-1, 1) @ W2.T  # (100, 8)
    dl_dpre = dl_dh * (1 - h**2)  # tanh derivative
    dl_dW1 = X.T @ dl_dpre
    dl_db1 = dl_dpre.sum(axis=0)

    # K-FAC: Kronecker factors for layer 1
    A1 = (X.T @ X) / len(y) + damping * np.eye(5)         # input covariance
    G1 = (dl_dpre.T @ dl_dpre) / len(y) + damping * np.eye(8)  # gradient covariance
    # Natural gradient update: dW = A^{-1} @ dW_flat @ G^{-1}
    A1_inv = np.linalg.inv(A1)
    G1_inv = np.linalg.inv(G1)
    nat_dW1 = A1_inv @ dl_dW1 @ G1_inv

    # K-FAC: Kronecker factors for layer 2
    A2 = (h.T @ h) / len(y) + damping * np.eye(8)
    G2_scalar = np.mean(dl_dlogits**2) + damping
    nat_dW2 = np.linalg.inv(A2) @ dl_dW2 / G2_scalar

    W1 -= 0.1 * nat_dW1
    b1 -= 0.1 * dl_db1
    W2 -= 0.1 * nat_dW2
    b2 -= 0.1 * dl_db2

print(f"Final loss: {loss:.4f}")
# K-FAC typically converges in fewer steps than SGD or Adam
