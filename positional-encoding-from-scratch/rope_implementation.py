"""Rotary Position Embeddings (RoPE) — used in LLaMA, Mistral, etc.

Encodes position by rotating query/key vectors in 2D subspaces.
Position information lives in the angle, not a separate vector.
"""
import numpy as np

def apply_rope(x, positions):
    """
    Apply Rotary Position Embeddings to a (seq_len, d) matrix.
    x: shape (seq_len, d) — the vectors to rotate (Q or K)
    positions: shape (seq_len,) — position index for each token
    """
    seq_len, d = x.shape
    assert d % 2 == 0, "Dimension must be even"

    # Frequencies: theta_i = 1 / 10000^(2i/d)
    i = np.arange(d // 2)
    theta = 1.0 / (10000 ** (2 * i / d))         # (d/2,)

    # Angles: position * frequency
    angles = positions[:, np.newaxis] * theta[np.newaxis, :]  # (seq_len, d/2)

    cos_a = np.cos(angles)  # (seq_len, d/2)
    sin_a = np.sin(angles)  # (seq_len, d/2)

    # Split x into pairs: even dims and odd dims
    x_even = x[:, 0::2]    # (seq_len, d/2) — first element of each pair
    x_odd  = x[:, 1::2]    # (seq_len, d/2) — second element of each pair

    # Apply 2D rotation to each pair
    x_rot_even = x_even * cos_a - x_odd * sin_a
    x_rot_odd  = x_even * sin_a + x_odd * cos_a

    # Interleave back
    x_rot = np.empty_like(x)
    x_rot[:, 0::2] = x_rot_even
    x_rot[:, 1::2] = x_rot_odd
    return x_rot

# Example: rotate an 8-dim vector at positions 0, 1, 2
x = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
              [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
              [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])

positions = np.array([0, 1, 2])
x_rot = apply_rope(x, positions)

print("Original (all identical):")
print(np.round(x[0], 3))

print("\nAfter RoPE (each row rotated by its position):")
for p in range(3):
    print(f"  Position {p}: {np.round(x_rot[p], 3)}")
