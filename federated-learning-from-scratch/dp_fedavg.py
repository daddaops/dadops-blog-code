"""
DP-FedAvg: Differential Privacy for Federated Learning

Clips per-client updates and adds calibrated Gaussian noise
for formal differential privacy guarantees.

Blog post: https://dadops.dev/blog/federated-learning-from-scratch/
"""
from setup import *

hospitals, X_all, y_all, true_w = make_hospitals()


def dp_fedavg(hospitals, rounds=30, local_epochs=5, lr=0.5,
              clip_norm=1.0, noise_scale=0.0):
    """DP-FedAvg: clipping + Gaussian noise for differential privacy."""
    w_global = np.zeros(2)
    history = []

    for t in range(rounds):
        clipped_updates = []
        for X_k, y_k in hospitals:
            w_local = w_global.copy()
            for _ in range(local_epochs):
                p = sigmoid(X_k @ w_local)
                grad = X_k.T @ (p - y_k) / len(y_k)
                w_local -= lr * grad

            # Clip update to bound sensitivity
            delta = w_local - w_global
            norm = np.linalg.norm(delta)
            if norm > clip_norm:
                delta = delta * clip_norm / norm
            clipped_updates.append(delta)

        # Average clipped updates, then add calibrated noise
        avg_delta = np.mean(clipped_updates, axis=0)
        if noise_scale > 0:
            noise = np.random.randn(2) * noise_scale * clip_norm / len(hospitals)
            avg_delta += noise

        w_global += avg_delta
        history.append(acc(X_all, y_all, w_global))

    return history


hist_no_dp = dp_fedavg(hospitals, noise_scale=0.0)   # no privacy
hist_mod_dp = dp_fedavg(hospitals, noise_scale=1.0)   # moderate
hist_hi_dp = dp_fedavg(hospitals, noise_scale=5.0)    # strong

print(f"No DP (epsilon=inf):  {hist_no_dp[-1]:.0%}")
print(f"Moderate (sigma=1.0): {hist_mod_dp[-1]:.0%}")
print(f"Strong (sigma=5.0):   {hist_hi_dp[-1]:.0%}")
