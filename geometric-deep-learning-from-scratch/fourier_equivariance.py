import numpy as np

def shift_matrix(n):
    """Cyclic shift matrix: moves element i to position (i+1) % n."""
    T = np.zeros((n, n))
    for i in range(n):
        T[(i + 1) % n, i] = 1.0
    return T

n = 8
T = shift_matrix(n)

# Start with a random weight matrix
W_random = np.random.randn(n, n)

# Project onto the space of matrices that commute with T.
# A matrix commutes with all cyclic shifts iff it is circulant.
# Circulant matrices are diagonal in the Fourier basis.
F = np.fft.fft(np.eye(n), axis=0) / np.sqrt(n)  # DFT matrix
F_inv = np.conj(F).T

# Project: zero out off-diagonal elements in Fourier domain
W_fourier = F @ W_random @ F_inv
W_diag = np.diag(np.diag(W_fourier))         # keep diagonal only
W_equivariant = np.real(F_inv @ W_diag @ F)  # back to spatial

# Verify: W_equivariant commutes with T
commutator = W_equivariant @ T - T @ W_equivariant
print(f"Max commutator error: {np.max(np.abs(commutator)):.2e}")  # ~0

# The equivariant matrix is circulant: each row is a shifted first row
print(f"First row (the kernel): {np.round(W_equivariant[0], 2)}")
print(f"Parameters: {n} (kernel) vs {n*n} (unconstrained)")
# Convolution IS the unique translation-equivariant linear map!
