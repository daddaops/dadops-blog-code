"""Shared helpers for quantization scripts."""
import numpy as np


def symmetric_quantize(weights, bits=8):
    """Quantize weights symmetrically around zero."""
    q_max = 2**(bits - 1) - 1
    alpha = np.max(np.abs(weights))
    scale = alpha / q_max
    quantized = np.clip(np.round(weights / scale), -q_max, q_max).astype(int)
    dequantized = scale * quantized.astype(float)
    return quantized, scale, dequantized
