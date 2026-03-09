"""
Temperature-Scaled Softmax

From: https://dadops.io/blog/softmax-temperature-from-scratch/

Covers:
- Softmax with temperature parameter (Boltzmann distribution)
- Temperature sweep showing how T affects distribution sharpness
"""

import numpy as np


def softmax_with_temperature(z, T=1.0):
    """Softmax with temperature scaling."""
    z_scaled = z / T                         # scale logits by temperature
    z_shifted = z_scaled - np.max(z_scaled)  # numerical stability
    e = np.exp(z_shifted)
    return e / np.sum(e)


if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5])
    tokens = ["mat", "rug", "floor", "carpet", "ground"]

    print("=== Temperature sweep ===")
    for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
        probs = softmax_with_temperature(logits, T)
        top = tokens[np.argmax(probs)]
        print(f"T={T:4.1f}  {np.array2string(probs, precision=3, floatmode='fixed')}"
              f"  top: {top} ({probs.max():.1%})")

    # T= 0.1  [1.000  0.000  0.000  0.000  0.000]  top: mat (100.0%)
    # T= 0.5  [0.826  0.112  0.041  0.015  0.006]  top: mat (82.6%)
    # T= 1.0  [0.553  0.203  0.123  0.075  0.045]  top: mat (55.3%)
    # T= 2.0  [0.366  0.222  0.173  0.135  0.105]  top: mat (36.6%)
    # T= 5.0  [0.261  0.213  0.193  0.175  0.158]  top: mat (26.1%)
