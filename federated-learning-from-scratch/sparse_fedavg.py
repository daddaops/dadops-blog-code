"""
FedAvg with Gradient Sparsification

Top-k sparsification with error feedback for bandwidth reduction.

Blog post: https://dadops.dev/blog/federated-learning-from-scratch/
"""
from setup import *

hospitals, X_all, y_all, true_w = make_hospitals()


def fedavg_sparse(hospitals, rounds=30, local_epochs=5, lr=0.5, top_k=1):
    """FedAvg with top-k sparsification and error feedback."""
    w_global = np.zeros(2)
    residuals = [np.zeros(2) for _ in hospitals]  # error feedback buffers
    bytes_sent = 0

    for t in range(rounds):
        updates, sizes = [], []
        for i, (X_k, y_k) in enumerate(hospitals):
            w_local = w_global.copy()
            for _ in range(local_epochs):
                p = sigmoid(X_k @ w_local)
                grad = X_k.T @ (p - y_k) / len(y_k)
                w_local -= lr * grad

            # Delta + accumulated residual from previous rounds
            delta = (w_local - w_global) + residuals[i]

            # Keep only top-k components by magnitude
            sparse = np.zeros_like(delta)
            top_idx = np.argsort(np.abs(delta))[-top_k:]
            sparse[top_idx] = delta[top_idx]

            residuals[i] = delta - sparse  # save what wasn't sent
            updates.append(sparse)
            sizes.append(len(y_k))
            bytes_sent += top_k * 8  # only k floats transmitted

        sizes = np.array(sizes, dtype=float)
        sizes /= sizes.sum()
        w_global += sum(s * u for s, u in zip(sizes, updates))

    final_acc = acc(X_all, y_all, w_global)
    return final_acc, bytes_sent


full_acc, full_bytes = fedavg_sparse(hospitals, top_k=2)  # full (2 params)
sp_acc, sp_bytes = fedavg_sparse(hospitals, top_k=1)      # 50% sparse

print(f"Full updates:  {full_acc:.0%} accuracy, {full_bytes:,} bytes")
print(f"50% sparse:    {sp_acc:.0%} accuracy, {sp_bytes:,} bytes")
