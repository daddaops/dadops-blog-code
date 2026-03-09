"""Loss landscape interpolation between two specialist models.

Demonstrates that linear interpolation between models trained from the
same initialization produces a U-shaped combined loss curve, with the
midpoint forming a generalist model.
"""
import numpy as np

def make_data(n, task):
    """Generate 2D classification data for two different tasks."""
    X = np.random.randn(n, 2)
    if task == 'horizontal':
        y = (X[:, 0] > 0).astype(float)  # classify by x-axis
    else:
        y = (X[:, 1] > 0).astype(float)  # classify by y-axis
    return X, y

def sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_net(X, y, W_init, lr=0.1, steps=300):
    """Train a single-layer network from given initial weights."""
    W = W_init.copy()
    for _ in range(steps):
        pred = sigmoid(X @ W)
        grad = X.T @ (pred - y.reshape(-1, 1)) / len(X)
        W -= lr * grad
    return W

def evaluate(X, y, W):
    """Binary cross-entropy loss."""
    pred = np.clip(sigmoid(X @ W), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(pred.flatten()) + (1-y) * np.log(1-pred.flatten()))

# Same initialization for both models (same "mountain lodge")
np.random.seed(42)
W_init = np.random.randn(2, 1) * 0.5

# Train on different tasks
X1, y1 = make_data(200, 'horizontal')
X2, y2 = make_data(200, 'vertical')
W_A = train_net(X1, y1, W_init)
W_B = train_net(X2, y2, W_init)

# Walk the interpolation path
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    W_merged = (1 - t) * W_A + t * W_B
    loss1 = evaluate(X1, y1, W_merged)
    loss2 = evaluate(X2, y2, W_merged)
    print(f"t={t:.2f}  Task1={loss1:.3f}  Task2={loss2:.3f}  Combined={loss1+loss2:.3f}")

# Output:
# t=0.00  Task1=0.193  Task2=0.726  Combined=0.919  (specialist A)
# t=0.25  Task1=0.247  Task2=0.561  Combined=0.808  (improving!)
# t=0.50  Task1=0.354  Task2=0.398  Combined=0.752  (sweet spot)
# t=0.75  Task1=0.530  Task2=0.283  Combined=0.813
# t=1.00  Task1=0.730  Task2=0.195  Combined=0.925  (specialist B)
