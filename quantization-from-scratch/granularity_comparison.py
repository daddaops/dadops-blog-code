"""Per-tensor, per-channel, per-group quantization comparison."""
import numpy as np
from helpers import symmetric_quantize

def quantize_per_tensor(W, bits=4):
    """One scale for the entire matrix."""
    _, scale, deq = symmetric_quantize(W.flatten(), bits)
    return deq.reshape(W.shape), 1

def quantize_per_channel(W, bits=4):
    """One scale per row (output channel)."""
    deq = np.zeros_like(W)
    for i in range(W.shape[0]):
        _, scale, deq[i] = symmetric_quantize(W[i], bits)
    return deq, W.shape[0]

def quantize_per_group(W, bits=4, group_size=128):
    """One scale per group of `group_size` elements in each row."""
    deq = np.zeros_like(W)
    num_scales = 0
    for i in range(W.shape[0]):
        for j in range(0, W.shape[1], group_size):
            chunk = W[i, j:j+group_size]
            _, scale, deq[i, j:j+group_size] = symmetric_quantize(chunk, bits)
            num_scales += 1
    return deq, num_scales

# Create a weight matrix with one outlier channel
np.random.seed(42)
W = np.random.randn(64, 512).astype(np.float32) * 0.02
W[13, :] *= 50  # Channel 13 has abnormally large weights

print("Quantization granularity comparison (INT4):")
print(f"{'Method':<16} {'MSE':>12} {'Scale factors':>14}")
print("-" * 44)

for name, fn in [("Per-tensor", quantize_per_tensor),
                 ("Per-channel", quantize_per_channel),
                 ("Per-group", lambda W, b: quantize_per_group(W, b, 128))]:
    deq, n_scales = fn(W, 4)
    mse = np.mean((W - deq) ** 2)
    print(f"{name:<16} {mse:>12.8f} {n_scales:>14}")
