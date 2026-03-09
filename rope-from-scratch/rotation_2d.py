"""2D rotation demonstrating RoPE's relative position property."""
import numpy as np


def rotate_2d(x, position, theta):
    """Rotate a 2D vector by angle position * theta."""
    cos_val = np.cos(position * theta)
    sin_val = np.sin(position * theta)
    return np.array([
        x[0] * cos_val - x[1] * sin_val,
        x[0] * sin_val + x[1] * cos_val
    ])

# Two content vectors (these stay fixed)
q = np.array([0.5, 0.8])
k = np.array([0.3, 0.6])
theta = 0.1

# Same relative offset (3), wildly different absolute positions
for m, n in [(2, 5), (10, 13), (100, 103), (9999, 10002)]:
    q_rot = rotate_2d(q, m, theta)
    k_rot = rotate_2d(k, n, theta)
    dot = np.dot(q_rot, k_rot)
    print(f"positions ({m:5d}, {n:5d}) -> dot product = {dot:.6f}")

# Output:
# positions (    2,     5) -> dot product = 0.584131
# positions (   10,    13) -> dot product = 0.584131
# positions (  100,   103) -> dot product = 0.584131
# positions ( 9999, 10002) -> dot product = 0.584131
