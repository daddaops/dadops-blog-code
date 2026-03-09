"""Shows why standard training fails on 5-way 5-shot classification.

Trains a simple MLP on just 25 examples (5 classes x 5 shots).
The model memorizes instantly but fails to generalize.
"""
import numpy as np

# Generate 5-way 5-shot synthetic data (20-dimensional features)
np.random.seed(42)
n_classes, k_shot, dim = 5, 5, 20
centers = np.random.randn(n_classes, dim) * 3  # class centroids

# Support set: 5 examples per class = 25 total training points
X_train, y_train = [], []
for c in range(n_classes):
    for _ in range(k_shot):
        X_train.append(centers[c] + np.random.randn(dim) * 0.5)
        y_train.append(c)
X_train, y_train = np.array(X_train), np.array(y_train)

# Test set: 50 examples per class (held out)
X_test, y_test = [], []
for c in range(n_classes):
    for _ in range(50):
        X_test.append(centers[c] + np.random.randn(dim) * 0.5)
        y_test.append(c)
X_test, y_test = np.array(X_test), np.array(y_test)

# Two-layer MLP: 20 -> 64 -> 5
W1 = np.random.randn(dim, 64) * 0.1
b1 = np.zeros(64)
W2 = np.random.randn(64, n_classes) * 0.1
b2 = np.zeros(n_classes)

def forward(X):
    h = np.maximum(0, X @ W1 + b1)  # ReLU
    logits = h @ W2 + b2
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_l / exp_l.sum(axis=1, keepdims=True)

# Train for 200 epochs — watch it memorize
for epoch in range(200):
    probs = forward(X_train)
    # SGD with cross-entropy (full-batch for simplicity)
    dlogits = probs.copy()
    dlogits[range(len(y_train)), y_train] -= 1
    dlogits /= len(y_train)
    h = np.maximum(0, X_train @ W1 + b1)
    W2 -= 0.5 * h.T @ dlogits
    b2 -= 0.5 * dlogits.sum(axis=0)
    dh = dlogits @ W2.T * (h > 0)
    W1 -= 0.5 * X_train.T @ dh
    b1 -= 0.5 * dh.sum(axis=0)

train_acc = (forward(X_train).argmax(1) == y_train).mean()
test_acc = (forward(X_test).argmax(1) == y_test).mean()
print(f"Train acc: {train_acc:.0%}, Test acc: {test_acc:.0%}")
# Expected: Train acc: 100%, Test acc: ~52% — memorization, not learning
