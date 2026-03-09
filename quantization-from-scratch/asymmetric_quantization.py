"""Asymmetric quantization for skewed distributions."""
import numpy as np
from helpers import symmetric_quantize

def asymmetric_quantize(weights, bits=8):
    """Quantize weights with a zero-point offset."""
    q_max = 2**bits - 1
    w_min, w_max = np.min(weights), np.max(weights)
    scale = (w_max - w_min) / q_max
    zero_point = int(np.round(-w_min / scale))
    quantized = np.clip(np.round(weights / scale) + zero_point,
                        0, q_max).astype(int)
    dequantized = scale * (quantized.astype(float) - zero_point)
    return quantized, scale, zero_point, dequantized

# Compare symmetric vs asymmetric on a skewed distribution
# (post-ReLU activations are always non-negative)
np.random.seed(42)
relu_activations = np.abs(np.random.randn(1000).astype(np.float32) * 0.5)

# Symmetric wastes the negative range
_, _, deq_sym = symmetric_quantize(relu_activations, bits=4)
mse_sym = np.mean((relu_activations - deq_sym) ** 2)

# Asymmetric uses the full range for positive values
_, _, _, deq_asym = asymmetric_quantize(relu_activations, bits=4)
mse_asym = np.mean((relu_activations - deq_asym) ** 2)

print(f"INT4 Symmetric  MSE: {mse_sym:.6f}")
print(f"INT4 Asymmetric MSE: {mse_asym:.6f}")
print(f"Improvement: {(1 - mse_asym/mse_sym)*100:.1f}%")
