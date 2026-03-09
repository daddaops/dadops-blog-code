"""
FedAvg Algorithm

Implements Federated Averaging (McMahan et al. 2017).
Each round: local training for E epochs, then weighted aggregation.

Blog post: https://dadops.dev/blog/federated-learning-from-scratch/
"""
from setup import *

hospitals, X_all, y_all, true_w = make_hospitals()
w_pooled = train(X_all, y_all)


def fedavg(hospitals, rounds=30, local_epochs=5, lr=0.5):
    """Federated Averaging — McMahan et al. 2017."""
    w_global = np.zeros(2)
    history = []

    for t in range(rounds):
        updates = []
        sizes = []
        for X_k, y_k in hospitals:
            w_local = w_global.copy()

            # Local training: E epochs of gradient descent
            for _ in range(local_epochs):
                p = sigmoid(X_k @ w_local)
                grad = X_k.T @ (p - y_k) / len(y_k)
                w_local -= lr * grad

            updates.append(w_local)
            sizes.append(len(y_k))

        # Weighted aggregation: each client weighted by dataset size
        sizes = np.array(sizes, dtype=float)
        sizes /= sizes.sum()
        w_global = sum(s * u for s, u in zip(sizes, updates))
        history.append(acc(X_all, y_all, w_global))

    return w_global, history


w_fed5, hist5 = fedavg(hospitals, rounds=30, local_epochs=5)
w_fed1, hist1 = fedavg(hospitals, rounds=30, local_epochs=1)  # FedSGD

print(f"FedAvg  (E=5, 30 rounds): {acc(X_all, y_all, w_fed5):.0%}")
print(f"FedSGD  (E=1, 30 rounds): {acc(X_all, y_all, w_fed1):.0%}")
print(f"Centralized (pooled):     {acc(X_all, y_all, w_pooled):.0%}")
