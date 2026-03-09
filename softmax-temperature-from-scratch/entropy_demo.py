"""
Entropy as a Function of Temperature

From: https://dadops.io/blog/softmax-temperature-from-scratch/

Covers:
- Shannon entropy measurement of distribution spread
- How entropy increases with temperature toward the uniform maximum
"""

import numpy as np


def softmax_with_temperature(z, T=1.0):
    """Softmax with temperature scaling."""
    z_scaled = z / T
    z_shifted = z_scaled - np.max(z_scaled)
    e = np.exp(z_shifted)
    return e / np.sum(e)


def entropy(probs):
    """Shannon entropy in bits."""
    # Filter out zeros to avoid log(0)
    p = probs[probs > 0]
    return -np.sum(p * np.log2(p))


if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5])

    print("=== Entropy vs Temperature ===")
    for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
        probs = softmax_with_temperature(logits, T)
        H = entropy(probs)
        print(f"T={T:4.1f}  entropy={H:.3f} bits  (max possible: {np.log2(5):.3f})")

    # T= 0.1  entropy=0.000 bits  (max possible: 2.322)
    # T= 0.5  entropy=0.906 bits  (max possible: 2.322)
    # T= 1.0  entropy=1.793 bits  (max possible: 2.322)
    # T= 2.0  entropy=2.182 bits  (max possible: 2.322)
    # T= 5.0  entropy=2.299 bits  (max possible: 2.322)
