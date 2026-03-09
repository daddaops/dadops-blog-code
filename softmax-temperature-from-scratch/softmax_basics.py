"""
Softmax Basics: Why We Need Softmax and How It Works

From: https://dadops.io/blog/softmax-temperature-from-scratch/

Covers:
- Why naive normalization fails for logits
- The softmax function from first principles
- Shift invariance property
"""

import numpy as np


def softmax_naive(z):
    """Softmax: exponentiate and normalize."""
    e = np.exp(z)
    return e / np.sum(e)


if __name__ == "__main__":
    # --- Why not just divide by the sum? ---
    print("=== Attempt 1: divide by sum ===")
    logits = np.array([2.0, 1.0, -1.0, 3.0, -0.5])

    probs = logits / np.sum(logits)
    print(probs)
    # [ 0.444  0.222 -0.222  0.667 -0.111]
    #   ^^ Negative "probabilities"! That's not a valid distribution.

    print("\n=== Attempt 2: absolute values then normalize ===")
    probs = np.abs(logits) / np.sum(np.abs(logits))
    print(probs)
    # [0.267  0.133  0.133  0.400  0.067]
    #   ^^ Looks plausible, but -1.0 and +1.0 get the same probability!
    #   We've lost all ordering information.

    print("\n=== Attempt 3: min-max scaling to [0, 1] ===")
    probs = (logits - logits.min()) / (logits.max() - logits.min())
    probs = probs / probs.sum()
    print(probs)
    # [0.316  0.211  0.000  0.421  0.053]  (approximately)
    #   ^^ Zero probability for the minimum! And breaks entirely
    #   when all logits are equal (division by zero).

    # --- Softmax from first principles ---
    print("\n=== Softmax (naive) ===")
    logits = np.array([2.0, 1.0, -1.0, 3.0, -0.5])
    probs = softmax_naive(logits)
    print(probs)
    # [0.237  0.087  0.012  0.645  0.019]
    print(f"Sum: {probs.sum():.6f}")
    # Sum: 1.000000

    # --- Step-by-step trace ---
    print("\n=== Step-by-step trace ===")
    z = np.array([5.0, 2.0, 1.0])

    # Step 1: exponentiate each
    exps = np.exp(z)
    print(f"exp(5.0) = {exps[0]:.2f}")   # 148.41
    print(f"exp(2.0) = {exps[1]:.2f}")   #   7.39
    print(f"exp(1.0) = {exps[2]:.2f}")   #   2.72

    # Step 2: sum them up
    total = np.sum(exps)
    print(f"Total:     {total:.2f}")      # 158.51

    # Step 3: divide each by total
    probs = exps / total
    print(f"Probabilities: {probs}")
    # [0.936  0.047  0.017]
    # The "5.0" logit captures 93.6% of the probability mass!

    # --- Shift invariance ---
    print("\n=== Shift invariance ===")
    z = np.array([5.0, 2.0, 1.0])
    print(softmax_naive(z))
    # [0.936  0.047  0.017]

    # Add 100 to everything
    print(softmax_naive(z + 100))
    # [0.936  0.047  0.017]  <- identical!

    # Subtract 5 from everything
    print(softmax_naive(z - 5))
    # [0.936  0.047  0.017]  <- still identical!
