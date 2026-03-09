import numpy as np

# --- C4: 90-degree rotations on a 4x4 image ---
image = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

rot90 = np.rot90(image, k=1)   # 90 degrees counterclockwise
rot180 = np.rot90(image, k=2)  # 180 degrees
rot270 = np.rot90(image, k=3)  # 270 degrees

# Group axioms: rot90 composed 4 times = identity
assert np.array_equal(np.rot90(image, k=4), image)

# --- S_n: permutations on a graph adjacency matrix ---
# Path graph: 0-1-2
A = np.array([[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]])

# Permutation: swap node 0 and node 2
P = np.array([[0, 0, 1],
              [0, 1, 0],
              [1, 0, 0]])

A_permuted = P @ A @ P.T  # Correct way to permute a graph
# Node features transform as: x' = P @ x
# Adjacency transforms as:    A' = P @ A @ P^T

print(f"Original A:\n{A}")
print(f"Permuted A:\n{A_permuted}")
# Structure preserved: same edges, different labeling
