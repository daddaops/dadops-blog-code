"""Residual block with projection shortcuts for dimension changes."""
import numpy as np
from helpers import conv2d

class ResidualBlockWithProjection:
    """Residual block that handles spatial downsampling and channel changes."""

    def __init__(self, in_channels, out_channels, stride, rng):
        scale1 = np.sqrt(2.0 / (in_channels * 9))
        scale2 = np.sqrt(2.0 / (out_channels * 9))
        self.W1 = rng.randn(out_channels, in_channels, 3, 3) * scale1
        self.W2 = rng.randn(out_channels, out_channels, 3, 3) * scale2
        self.stride = stride

        # Projection shortcut: 1x1 conv when dimensions change
        self.needs_projection = (stride != 1) or (in_channels != out_channels)
        if self.needs_projection:
            scale_proj = np.sqrt(2.0 / in_channels)
            self.W_proj = rng.randn(out_channels, in_channels, 1, 1) * scale_proj

    def forward(self, x):
        identity = x

        # Main path: conv(stride) -> BN -> ReLU -> conv -> BN
        out = conv2d(x, self.W1, stride=self.stride, pad=1)
        out = np.maximum(0, out)  # (BN omitted for clarity)
        out = conv2d(out, self.W2, stride=1, pad=1)

        # Shortcut path: project if dimensions changed
        if self.needs_projection:
            identity = conv2d(x, self.W_proj, stride=self.stride, pad=0)

        # Add and activate
        out = out + identity  # Shapes now match!
        out = np.maximum(0, out)
        return out

# Example: transition from stage 1 (64 channels) to stage 2 (128 channels)
# Input:  (64, 32, 32)  -- 64 channels, 32x32 spatial
# Output: (128, 16, 16) -- 128 channels, 16x16 spatial (stride 2)
# The 1x1 projection maps: (64, 32, 32) -> (128, 16, 16)
# The main path maps:       (64, 32, 32) -> (128, 16, 16) via stride-2 conv

# Smoke test with small dimensions
rng = np.random.RandomState(42)
block = ResidualBlockWithProjection(in_channels=4, out_channels=8, stride=2, rng=rng)
x = rng.randn(4, 8, 8)
out = block.forward(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Projection used: {block.needs_projection}")
