"""
Attention Scaling as Temperature

From: https://dadops.io/blog/softmax-temperature-from-scratch/

Covers:
- The sqrt(d_k) scaling factor in attention IS a temperature parameter
- Connection between attention dimension and effective temperature
"""

import numpy as np


if __name__ == "__main__":
    # The attention scaling factor IS a temperature
    d_k = 64           # typical head dimension
    T_effective = np.sqrt(d_k)  # = 8.0

    print(f"Head dimension d_k = {d_k}")
    print(f"Effective temperature T = sqrt({d_k}) = {T_effective:.1f}")
    print()
    print("This is equivalent to:")
    print(f"  softmax(QK^T / {T_effective:.1f}) = softmax_with_temperature(QK^T, T={T_effective:.1f})")
