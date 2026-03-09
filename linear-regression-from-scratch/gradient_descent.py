import numpy as np

def gradient_descent_lr(X, y, lr=0.01, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []

    for epoch in range(epochs):
        y_pred = X @ w + b
        residual = y_pred - y

        grad_w = (2 / n) * (X.T @ residual)
        grad_b = (2 / n) * np.sum(residual)

        w -= lr * grad_w
        b -= lr * grad_b

        loss = np.mean(residual ** 2)
        losses.append(loss)
    return w, b, losses

# Generate data
np.random.seed(42)
n, d = 500, 5
true_w = np.array([3.0, -1.5, 0.0, 2.0, 0.0])
X_raw = np.random.randn(n, d) * np.array([1, 10, 0.1, 5, 100])  # different scales
y = X_raw @ true_w + 4.0 + np.random.randn(n) * 0.5

# Without scaling — slow, zigzaggy convergence
w_raw, b_raw, loss_raw = gradient_descent_lr(X_raw, y, lr=0.00001, epochs=2000)

# With scaling — fast, direct convergence
X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
X_scaled = (X_raw - X_mean) / X_std
w_scaled, b_scaled, loss_scaled = gradient_descent_lr(X_scaled, y, lr=0.1, epochs=200)

# Convert scaled weights back to original space
w_original = w_scaled / X_std
b_original = b_scaled - np.sum(w_scaled * X_mean / X_std)

print("True weights:       ", true_w)
print("GD (no scaling):    ", np.round(w_raw, 3))
print("GD (with scaling):  ", np.round(w_original, 3))
print(f"\nLoss after 2000 epochs (no scaling):  {loss_raw[-1]:.4f}")
print(f"Loss after 200 epochs (with scaling): {loss_scaled[-1]:.4f}")
print("Scaling wins: 10x fewer epochs, 10,000x larger learning rate")
