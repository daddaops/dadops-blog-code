"""L1 and L2 regularization comparison."""
import numpy as np

# Generate data with same seed as blog block 1
np.random.seed(42)
X_train = np.random.uniform(-3, 3, (50, 1))
y_train = np.sin(X_train) + np.random.randn(50, 1) * 0.3
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
y_test = np.sin(X_test)

def train_with_penalty(penalty_type, lam, epochs=3000):
    np.random.seed(42)
    dims = [1, 64, 64, 64, 1]
    W = [np.random.randn(dims[i], dims[i+1]) * 0.5 for i in range(4)]
    b = [np.zeros((1, dims[i+1])) for i in range(4)]

    for epoch in range(epochs):
        h = X_train
        acts = [h]
        for i in range(3):
            h = np.maximum(0, h @ W[i] + b[i])
            acts.append(h)
        out = h @ W[3] + b[3]

        grad = 2 * (out - y_train) / len(y_train)
        for i in range(3, -1, -1):
            dW = acts[i].T @ grad
            if penalty_type == "l2":
                dW += 2 * lam * W[i]
            elif penalty_type == "l1":
                dW += lam * np.sign(W[i])
            W[i] -= 0.005 * dW
            b[i] -= 0.005 * grad.sum(axis=0, keepdims=True)
            if i > 0:
                grad = (grad @ W[i].T) * (acts[i] > 0)

    total_w = sum(w.size for w in W)
    zeros = sum(np.sum(np.abs(w) < 1e-6) for w in W)
    h = X_test
    for i in range(3):
        h = np.maximum(0, h @ W[i] + b[i])
    test_loss = np.mean((h @ W[3] + b[3] - y_test) ** 2)
    return test_loss, zeros, total_w

none_loss, _, _ = train_with_penalty(None, 0)
l2_loss, l2_z, total = train_with_penalty("l2", 0.01)
l1_loss, l1_z, total = train_with_penalty("l1", 0.001)

print(f"No reg   — test loss: {none_loss:.4f}, zeros: 0/{total}")
print(f"L2       — test loss: {l2_loss:.4f},  zeros: {l2_z}/{total}")
print(f"L1       — test loss: {l1_loss:.4f},  zeros: {l1_z}/{total}")
