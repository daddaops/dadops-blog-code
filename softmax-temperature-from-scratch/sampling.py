"""
Sampling Strategies: Top-k, Top-p (Nucleus), and Temperature Interaction

From: https://dadops.io/blog/softmax-temperature-from-scratch/

Covers:
- Top-k filtering
- Top-p (nucleus) filtering (Holtzman et al., 2020)
- How temperature affects nucleus size
"""

import numpy as np


def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z)
    e = np.exp(z_shifted)
    return e / np.sum(e)


def top_k_filter(logits, k):
    """Zero out all logits except the top k."""
    if k >= len(logits):
        return logits.copy()
    indices = np.argsort(logits)[-k:]   # indices of top k
    filtered = np.full_like(logits, -np.inf)
    filtered[indices] = logits[indices]
    return filtered


def top_p_filter(logits, p):
    """Keep the smallest set of tokens with cumulative probability >= p."""
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]   # highest prob first
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)

    # Find cutoff: first index where cumulative >= p
    cutoff_idx = np.searchsorted(cumulative, p) + 1
    keep_indices = sorted_indices[:cutoff_idx]

    filtered = np.full_like(logits, -np.inf)
    filtered[keep_indices] = logits[keep_indices]
    return filtered


if __name__ == "__main__":
    # --- Top-k sampling ---
    print("=== Top-k Sampling (k=3) ===")
    logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, -0.5, -1.0])
    tokens = ["mat", "rug", "floor", "carpet", "ground", "tile", "dirt", "mud"]

    filtered = top_k_filter(logits, k=3)
    probs = softmax(filtered)
    for tok, p in zip(tokens, probs):
        if p > 0.001:
            print(f"  {tok:8s} {p:.1%}")

    # mat       50.7%
    # rug       30.7%
    # floor     18.6%

    # --- Top-p (nucleus) sampling ---
    print("\n=== Top-p Sampling (p=0.9) ===")
    logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, -0.5, -1.0])
    tokens = ["mat", "rug", "floor", "carpet", "ground", "tile", "dirt", "mud"]

    filtered = top_p_filter(logits, p=0.9)
    probs = softmax(filtered)
    print("Top-p=0.9 nucleus:")
    for tok, p_val in zip(tokens, probs):
        if p_val > 0.001:
            print(f"  {tok:8s} {p_val:.1%}")

    # Top-p=0.9 nucleus:
    # mat       42.9%
    # rug       26.0%
    # floor     15.8%
    # carpet     9.6%
    # ground     5.8%
    # (5 tokens capture >90% of the mass)

    # --- Temperature affects nucleus size ---
    print("\n=== Temperature vs Nucleus Size ===")
    logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, -0.5, -1.0])

    for T in [0.3, 1.0, 2.0]:
        scaled = logits / T
        probs = softmax(scaled)
        sorted_probs = np.sort(probs)[::-1]
        cumulative = np.cumsum(sorted_probs)
        nucleus_size = np.searchsorted(cumulative, 0.9) + 1
        print(f"T={T:.1f}: nucleus size for p=0.9 is {nucleus_size} tokens")

    # T=0.3: nucleus size for p=0.9 is 2 tokens
    # T=1.0: nucleus size for p=0.9 is 5 tokens
    # T=2.0: nucleus size for p=0.9 is 6 tokens
