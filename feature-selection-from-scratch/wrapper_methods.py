"""
Wrapper Methods: Forward Selection and RFE

Implements forward selection (greedy addition) and Recursive Feature
Elimination (greedy removal) using logistic regression.

Blog post: https://dadops.dev/blog/feature-selection-from-scratch/
"""
import numpy as np
from dataset import make_dataset, sigmoid

X, y, names, n = make_dataset()


def logistic_accuracy(X_train, y_train, X_val, y_val, lr=0.1, epochs=200):
    """Train logistic regression, return validation accuracy."""
    w = np.zeros(X_train.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(X_train @ w + b)
        grad_w = X_train.T @ (p - y_train) / len(y_train)
        grad_b = (p - y_train).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    preds = (sigmoid(X_val @ w + b) > 0.5).astype(int)
    return (preds == y_val).mean(), np.abs(w)


# Train/validation split
idx = np.arange(n)
np.random.shuffle(idx)
Xtr, ytr = X[idx[:140]], y[idx[:140]]
Xval, yval = X[idx[140:]], y[idx[140:]]

# --- Forward Selection ---
selected_fwd = []
remaining_fwd = list(range(10))
acc_curve_fwd = []

for step in range(6):
    best_acc, best_f = -1, None
    for f in remaining_fwd:
        cols = selected_fwd + [f]
        acc, _ = logistic_accuracy(Xtr[:, cols], ytr, Xval[:, cols], yval)
        if acc > best_acc:
            best_acc, best_f = acc, f
    selected_fwd.append(best_f)
    remaining_fwd.remove(best_f)
    acc_curve_fwd.append(best_acc)

print("Forward Selection path:")
for i, f in enumerate(selected_fwd):
    print(f"  Step {i+1}: +{names[f]:<10s} acc = {acc_curve_fwd[i]:.3f}")

# --- Recursive Feature Elimination ---
active_rfe = list(range(10))
elim_order = []

while len(active_rfe) > 3:
    _, importances = logistic_accuracy(Xtr[:, active_rfe], ytr,
                                       Xval[:, active_rfe], yval)
    least = np.argmin(importances)
    elim_order.append(names[active_rfe[least]])
    active_rfe.pop(least)

print("\nRFE elimination order:", elim_order)
print("RFE surviving features:", [names[i] for i in active_rfe])
