"""Batch normalization for convolutional layers from scratch."""
import numpy as np

def batch_norm_2d(x_batch, gamma, beta, eps=1e-5):
    """Batch normalization for convolutional layers.
    x_batch: (N, C, H, W) -- batch of feature maps
    gamma:   (C,) -- learnable scale per channel
    beta:    (C,) -- learnable shift per channel
    returns: (N, C, H, W) -- normalized feature maps
    """
    n, c, h, w = x_batch.shape

    # Mean and variance per channel, across batch + spatial dims
    mean = x_batch.mean(axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1)
    var  = x_batch.var(axis=(0, 2, 3), keepdims=True)     # (1, C, 1, 1)

    # Normalize
    x_norm = (x_batch - mean) / np.sqrt(var + eps)

    # Scale and shift (reshape gamma/beta to broadcast)
    gamma = gamma.reshape(1, c, 1, 1)
    beta = beta.reshape(1, c, 1, 1)

    return gamma * x_norm + beta

if __name__ == "__main__":
    # Example: batch of 4 images, 6 feature maps each
    batch = np.random.randn(4, 6, 12, 12)
    gamma = np.ones(6)    # initialized to 1 (identity scaling)
    beta = np.zeros(6)    # initialized to 0 (no shift)

    normed = batch_norm_2d(batch, gamma, beta)
    print("Channel means after BN:", normed.mean(axis=(0,2,3)).round(4))  # ~0
    print("Channel stds after BN:", normed.std(axis=(0,2,3)).round(4))   # ~1
