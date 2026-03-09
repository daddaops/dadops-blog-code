"""Prove RoPE dot product depends on relative position only.

Same query-key pair at different absolute positions but same offset
produces identical dot products.
"""
import numpy as np

def apply_rope(x, positions):
    """Apply Rotary Position Embeddings to a (seq_len, d) matrix."""
    seq_len, d = x.shape
    assert d % 2 == 0
    i = np.arange(d // 2)
    theta = 1.0 / (10000 ** (2 * i / d))
    angles = positions[:, np.newaxis] * theta[np.newaxis, :]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    x_even = x[:, 0::2]
    x_odd  = x[:, 1::2]
    x_rot_even = x_even * cos_a - x_odd * sin_a
    x_rot_odd  = x_even * sin_a + x_odd * cos_a
    x_rot = np.empty_like(x)
    x_rot[:, 0::2] = x_rot_even
    x_rot[:, 1::2] = x_rot_odd
    return x_rot

# Prove: RoPE dot product depends on RELATIVE position only
np.random.seed(7)
d = 8
q = np.random.randn(1, d)  # A query vector
k = np.random.randn(1, d)  # A key vector

# Case 1: query at position 2, key at position 5 (offset = 3)
q_rot_2 = apply_rope(q, np.array([2]))
k_rot_5 = apply_rope(k, np.array([5]))
dot_case1 = (q_rot_2 @ k_rot_5.T)[0, 0]

# Case 2: query at position 10, key at position 13 (offset = 3)
q_rot_10 = apply_rope(q, np.array([10]))
k_rot_13 = apply_rope(k, np.array([13]))
dot_case2 = (q_rot_10 @ k_rot_13.T)[0, 0]

# Case 3: query at position 100, key at position 103 (offset = 3)
q_rot_100 = apply_rope(q, np.array([100]))
k_rot_103 = apply_rope(k, np.array([103]))
dot_case3 = (q_rot_100 @ k_rot_103.T)[0, 0]

print(f"Positions (2, 5),   offset 3: dot = {dot_case1:.6f}")
print(f"Positions (10, 13), offset 3: dot = {dot_case2:.6f}")
print(f"Positions (100,103),offset 3: dot = {dot_case3:.6f}")
print(f"All equal: {np.allclose(dot_case1, dot_case2) and np.allclose(dot_case2, dot_case3)}")
