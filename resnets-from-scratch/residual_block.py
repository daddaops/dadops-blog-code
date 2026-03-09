"""Basic residual block with 3x3 convolutions and batch norm."""
import numpy as np
from helpers import conv2d_same, batch_norm

class ResidualBlock:
    """A basic residual block: two conv layers with batch norm and a skip connection."""

    def __init__(self, channels, rng):
        # He initialization for both 3x3 conv layers
        scale = np.sqrt(2.0 / (channels * 9))  # fan_in = channels * 3 * 3
        self.W1 = rng.randn(channels, channels, 3, 3) * scale
        self.W2 = rng.randn(channels, channels, 3, 3) * scale
        # Batch norm parameters (simplified: just scale and shift)
        self.gamma1 = np.ones(channels)
        self.beta1 = np.zeros(channels)
        self.gamma2 = np.ones(channels)
        self.beta2 = np.zeros(channels)

    def forward(self, x):
        """x shape: (channels, H, W)"""
        identity = x  # Save for skip connection

        # First conv + BN + ReLU
        out = conv2d_same(x, self.W1)
        out = batch_norm(out, self.gamma1, self.beta1)
        out = np.maximum(0, out)  # ReLU

        # Second conv + BN
        out = conv2d_same(out, self.W2)
        out = batch_norm(out, self.gamma2, self.beta2)

        # Skip connection: add identity, then ReLU
        out = out + identity
        out = np.maximum(0, out)

        return out

# Smoke test: forward pass through a residual block
rng = np.random.RandomState(42)
channels = 4
block = ResidualBlock(channels, rng)
x = rng.randn(channels, 8, 8)
out = block.forward(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Shapes match: {x.shape == out.shape}")
