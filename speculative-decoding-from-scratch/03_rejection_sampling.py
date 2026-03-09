"""Rejection sampling mechanism for speculative decoding."""

import numpy as np


def rejection_sample(p, q, draft_token):
    """
    Speculative decoding rejection sampling for one position.

    p: target distribution (array over vocabulary)
    q: draft distribution (array over vocabulary)
    draft_token: index sampled from q

    Returns: (accepted: bool, output_token: int)
    """
    # Accept with probability min(1, p/q) for the drafted token
    accept_prob = min(1.0, p[draft_token] / q[draft_token])

    if np.random.random() < accept_prob:
        return True, draft_token

    # Rejected -- sample from residual distribution
    residual = np.maximum(0, p - q)
    residual /= residual.sum()
    correction = np.random.choice(len(p), p=residual)
    return False, correction


if __name__ == "__main__":
    np.random.seed(42)

    # Example: vocab = ["the", "cat", "sat", "dog"]
    p = np.array([0.50, 0.20, 0.10, 0.20])  # target
    q = np.array([0.40, 0.30, 0.20, 0.10])  # draft

    print("Running 10000 trials of rejection sampling for draft_token='cat' (idx 1)...")
    accept_count = 0
    token_counts = np.zeros(4)
    N = 10000
    for _ in range(N):
        accepted, token = rejection_sample(p, q, draft_token=1)
        if accepted:
            accept_count += 1
        token_counts[token] += 1

    words = ["the", "cat", "sat", "dog"]
    print(f"\nAcceptance rate: {accept_count/N:.3f} (expected ~0.667)")
    print("\nEmpirical vs target distribution:")
    for i, w in enumerate(words):
        print(f"  '{w}': empirical={token_counts[i]/N:.3f}, target={p[i]:.3f}")
