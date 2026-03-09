"""
Secure Aggregation

Pairwise masking so the server sees only the aggregate, not
individual client updates.

Blog post: https://dadops.dev/blog/federated-learning-from-scratch/
"""
import numpy as np


def secure_aggregate(client_updates, seed=42):
    """Simplified secure aggregation with pairwise masking."""
    K = len(client_updates)
    dim = len(client_updates[0])

    # Each client's masked update starts as its true update
    masked = [u.copy() for u in client_updates]
    rng = np.random.RandomState(seed)

    for i in range(K):
        for j in range(i + 1, K):
            # Clients i,j agree on a random mask (via shared PRNG seed)
            mask = rng.randn(dim) * 10  # large random noise
            masked[i] += mask   # client i adds mask
            masked[j] -= mask   # client j subtracts mask

    print("Individual masked updates (appear random):")
    for i, m in enumerate(masked):
        print(f"  Client {i}: [{m[0]:+7.2f}, {m[1]:+7.2f}]")

    true_agg = sum(client_updates)
    masked_agg = sum(masked)
    print(f"\nTrue aggregate:   [{true_agg[0]:.4f}, {true_agg[1]:.4f}]")
    print(f"Masked aggregate: [{masked_agg[0]:.4f}, {masked_agg[1]:.4f}]")
    print(f"Exact match: {np.allclose(true_agg, masked_agg)}")


# 4 clients with small, sensitive model updates
updates = [np.array([0.12, -0.05]), np.array([-0.08, 0.15]),
           np.array([0.05, 0.03]),  np.array([-0.03, -0.10])]
secure_aggregate(updates)
