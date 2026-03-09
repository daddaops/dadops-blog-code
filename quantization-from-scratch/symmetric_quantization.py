"""Symmetric quantization at different bit widths."""
import numpy as np
from helpers import symmetric_quantize

# Try it on a sample weight tensor
np.random.seed(42)
weights = np.random.randn(1000).astype(np.float32) * 0.5

for bits in [8, 4, 2]:
    q, scale, deq = symmetric_quantize(weights, bits)
    mse = np.mean((weights - deq) ** 2)
    max_err = np.max(np.abs(weights - deq))
    unique_levels = 2**bits
    print(f"INT{bits}: MSE={mse:.6f}, Max Error={max_err:.4f}, "
          f"Levels={unique_levels}, Compression={32//bits}x")
