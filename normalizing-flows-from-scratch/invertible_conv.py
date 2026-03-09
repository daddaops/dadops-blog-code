"""Invertible 1x1 convolutions (Glow).

Implements 1x1 invertible convolutions using LU decomposition
for efficient log-determinant computation.
"""
import numpy as np


def invertible_1x1_conv(channels=4, seed=42):
    """Implement a 1x1 invertible convolution with LU decomposition."""
    rng = np.random.RandomState(seed)

    # Initialize with a random orthogonal matrix
    W, _ = np.linalg.qr(rng.randn(channels, channels))

    # LU decompose: W = P @ L @ U
    from scipy.linalg import lu
    _, _, U = lu(W)  # W = P @ L @ U; det(P)=±1, det(L)=1

    # Log-determinant = sum of log|diagonal of U|
    log_det = np.sum(np.log(np.abs(np.diag(U))))

    # Forward: multiply channels by W at each spatial location
    # For a feature map of shape (H, W, C): output = input @ W.T
    H, W_spatial = 8, 8
    x = rng.randn(H, W_spatial, channels)
    z = x @ W.T  # forward pass

    # Inverse: multiply by W^(-1) = U^(-1) @ L^(-1) @ P.T
    W_inv = np.linalg.inv(W)
    x_recovered = z @ W_inv.T

    print(f"Weight matrix W shape: {W.shape}")
    print(f"Log|det W| per spatial location: {log_det:.4f}")
    print(f"Total log-det for {H}x{W_spatial} feature map: {H * W_spatial * log_det:.4f}")
    print(f"Reconstruction error: {np.max(np.abs(x - x_recovered)):.2e}")


if __name__ == "__main__":
    invertible_1x1_conv()
