"""Parallel prefix scan vs sequential scan for linear recurrences.

Demonstrates that the associative operator (a1,b1) ∘ (a2,b2) = (a2*a1, a2*b1+b2)
enables solving x_k = a_k*x_{k-1} + b_k in O(log L) parallel depth
instead of O(L) sequential steps.
"""
import numpy as np

def sequential_scan(a, b):
    """Sequential: x_k = a_k * x_{k-1} + b_k, with x_{-1} = 0"""
    L = len(a)
    x = np.zeros(L)
    x[0] = b[0]
    for k in range(1, L):
        x[k] = a[k] * x[k-1] + b[k]
    return x

def parallel_scan(a, b):
    """Parallel prefix scan using the associative operator:
    (a1, b1) ∘ (a2, b2) = (a2*a1, a2*b1 + b2)
    Solves x_k = a_k*x_{k-1} + b_k in O(log L) parallel depth."""
    L = len(a)
    aa, bb = a.copy(), b.copy()

    for d in range(int(np.ceil(np.log2(L)))):
        stride = 1 << d   # 1, 2, 4, 8, ...
        aa_prev, bb_prev = aa.copy(), bb.copy()
        for i in range(stride, L):
            # Combine element (i-stride) into element i
            bb[i] = aa_prev[i] * bb_prev[i - stride] + bb_prev[i]
            aa[i] = aa_prev[i] * aa_prev[i - stride]

    return bb  # bb[k] = x_k

# Verify
L = 16
np.random.seed(7)
a = np.random.uniform(0.8, 0.99, L)   # decay coefficients
b = np.random.randn(L) * 0.3          # inputs

x_seq = sequential_scan(a, b)
x_par = parallel_scan(a, b)

print(f"Sequential: {np.round(x_seq[:6], 4)}")
print(f"Parallel:   {np.round(x_par[:6], 4)}")
print(f"Max error:  {np.max(np.abs(x_seq - x_par)):.2e}")
print(f"Depth: O(log {L}) = {int(np.ceil(np.log2(L)))} steps vs {L} sequential")
