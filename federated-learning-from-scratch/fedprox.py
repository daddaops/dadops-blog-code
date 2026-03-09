"""
FedProx for Non-IID Data

Handles heterogeneous data distributions with a proximal term
that prevents local models from drifting too far from the global model.

Blog post: https://dadops.dev/blog/federated-learning-from-scratch/
"""
from setup import *

hospitals, X_all, y_all, true_w = make_hospitals()

# Create non-IID hospitals: skew each hospital's label distribution
noniid_hospitals = []
for i in range(5):
    r = np.random.RandomState(i + 100)
    X = r.randn(40, 2)
    p_true = sigmoid(X @ true_w)
    # Skew: hospitals 0-1 mostly class 0, hospitals 3-4 mostly class 1
    bias = (i - 2) * 0.7
    y = np.clip((p_true + bias > r.random(40)).astype(float), 0, 1)
    noniid_hospitals.append((X, y))


def fedprox(hospitals, rounds=30, local_epochs=5, lr=0.5, mu=0.0):
    """FedAvg with optional FedProx proximal term (mu=0 is plain FedAvg)."""
    w_global = np.zeros(2)
    history = []

    for t in range(rounds):
        updates, sizes = [], []
        for X_k, y_k in hospitals:
            w_local = w_global.copy()
            for _ in range(local_epochs):
                p = sigmoid(X_k @ w_local)
                grad = X_k.T @ (p - y_k) / len(y_k)
                grad += mu * (w_local - w_global)  # proximal term
                w_local -= lr * grad
            updates.append(w_local)
            sizes.append(len(y_k))

        sizes = np.array(sizes, dtype=float)
        sizes /= sizes.sum()
        w_global = sum(s * u for s, u in zip(sizes, updates))
        history.append(acc(X_all, y_all, w_global))

    return history


hist_avg = fedprox(noniid_hospitals, mu=0.0)   # plain FedAvg
hist_prox = fedprox(noniid_hospitals, mu=0.1)  # FedProx

print(f"Non-IID FedAvg:  final acc = {hist_avg[-1]:.0%}")
print(f"Non-IID FedProx: final acc = {hist_prox[-1]:.0%}")
