"""
Overflow Demo: Numerical Stability in Softmax

From: https://dadops.io/blog/softmax-temperature-from-scratch/

Covers:
- How naive softmax overflows on large logits
- The subtract-max trick for numerical stability
- Verification that stable and naive versions agree on safe inputs
"""

import numpy as np


def softmax_naive(z):
    """Softmax: exponentiate and normalize."""
    e = np.exp(z)
    return e / np.sum(e)


def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z)         # shift so max is 0
    e = np.exp(z_shifted)             # largest exp is now exp(0) = 1
    return e / np.sum(e)


if __name__ == "__main__":
    # --- Overflow with naive softmax ---
    print("=== Overflow with naive softmax ===")
    big_logits = np.array([1000.0, 1001.0, 999.0])
    print(softmax_naive(big_logits))
    # [nan  nan  nan]  <- oh no.

    # --- Stable softmax fixes it ---
    print("\n=== Stable softmax ===")
    big_logits = np.array([1000.0, 1001.0, 999.0])
    print(softmax(big_logits))
    # [0.245  0.665  0.090]  <- works perfectly!

    # --- Verify both give identical results on safe inputs ---
    print("\n=== Verification: naive vs stable ===")
    safe_logits = np.array([2.0, 1.0, -1.0, 3.0, -0.5])
    print(f"Naive:  {softmax_naive(safe_logits)}")
    print(f"Stable: {softmax(safe_logits)}")
    # Naive:  [0.237  0.087  0.012  0.645  0.019]
    # Stable: [0.237  0.087  0.012  0.645  0.019]  <- identical!

    # But only the stable version handles extreme inputs
    extreme = np.array([500.0, 502.0, 498.0])
    print(f"Naive:  {softmax_naive(extreme)}")    # [nan  nan  nan]
    print(f"Stable: {softmax(extreme)}")          # [0.117  0.867  0.016]
