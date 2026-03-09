"""Bottleneck block: 1x1 reduce -> 3x3 process -> 1x1 expand + skip."""
import numpy as np
from helpers import conv2d

class BottleneckBlock:
    """ResNet bottleneck: 1x1 reduce -> 3x3 process -> 1x1 expand + skip."""

    def __init__(self, in_channels, bottleneck_channels, out_channels, stride, rng):
        # 1x1 reduce
        s1 = np.sqrt(2.0 / in_channels)
        self.W_reduce = rng.randn(bottleneck_channels, in_channels, 1, 1) * s1
        # 3x3 process
        s2 = np.sqrt(2.0 / (bottleneck_channels * 9))
        self.W_process = rng.randn(bottleneck_channels, bottleneck_channels, 3, 3) * s2
        # 1x1 expand
        s3 = np.sqrt(2.0 / bottleneck_channels)
        self.W_expand = rng.randn(out_channels, bottleneck_channels, 1, 1) * s3

        self.stride = stride
        self.needs_projection = (stride != 1) or (in_channels != out_channels)
        if self.needs_projection:
            sp = np.sqrt(2.0 / in_channels)
            self.W_proj = rng.randn(out_channels, in_channels, 1, 1) * sp

    def forward(self, x):
        identity = x

        # Reduce: 256 -> 64
        out = conv2d(x, self.W_reduce, stride=1, pad=0)
        out = np.maximum(0, out)

        # Process at bottleneck width: 64 -> 64
        out = conv2d(out, self.W_process, stride=self.stride, pad=1)
        out = np.maximum(0, out)

        # Expand: 64 -> 256
        out = conv2d(out, self.W_expand, stride=1, pad=0)

        # Projection shortcut if needed
        if self.needs_projection:
            identity = conv2d(x, self.W_proj, stride=self.stride, pad=0)

        return np.maximum(0, out + identity)

# Smoke test with small dimensions
rng = np.random.RandomState(42)
block = BottleneckBlock(in_channels=16, bottleneck_channels=4, out_channels=16,
                        stride=1, rng=rng)
x = rng.randn(16, 8, 8)
out = block.forward(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Bottleneck ratio: {16}/{4} = {16//4}x reduction")
