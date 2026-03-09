"""d-dimensional RoPE and complex number equivalence."""
import numpy as np
from helpers import rope_frequencies, apply_rope


def apply_rope_complex(x, position, freqs):
    """Apply RoPE using complex arithmetic — equivalent to rotation matrix."""
    d = x.shape[-1]
    x_complex = x[..., :d//2] + 1j * x[..., d//2:]
    angles = position * freqs
    rotation = np.exp(1j * angles)
    x_rotated = x_complex * rotation
    return np.concatenate([x_rotated.real, x_rotated.imag], axis=-1)


# Verify: relative position property holds in d dimensions
d_model = 64
freqs = rope_frequencies(d_model)
q = np.random.randn(d_model)
k = np.random.randn(d_model)

for m, n in [(2, 5), (50, 53), (1000, 1003)]:
    q_rot = apply_rope(q, m, freqs)
    k_rot = apply_rope(k, n, freqs)
    print(f"positions ({m:4d}, {n:4d}) -> dot = {np.dot(q_rot, k_rot):.6f}")

# All three outputs are identical (relative offset = 3)

# Verify: complex version gives identical results to rotation matrix version
x = np.random.randn(d_model)
pos = 42

result_matrix = apply_rope(x, pos, freqs)
result_complex = apply_rope_complex(x, pos, freqs)
print(f"\nMax difference: {np.max(np.abs(result_matrix - result_complex)):.2e}")
# Output: Max difference: 0.00e+00
