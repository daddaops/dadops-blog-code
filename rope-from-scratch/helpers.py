"""Shared helpers for RoPE scripts."""
import numpy as np


def rope_frequencies(d_model, base=10000.0):
    """Compute rotation frequencies for each dimension pair."""
    i = np.arange(0, d_model, 2, dtype=np.float64)
    return 1.0 / (base ** (i / d_model))


def apply_rope(x, position, freqs):
    """Apply RoPE to a d-dimensional vector at a given position."""
    d = x.shape[-1]
    angles = position * freqs
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return np.concatenate([
        x1 * cos_a - x2 * sin_a,
        x2 * cos_a + x1 * sin_a
    ], axis=-1)


def softmax(x):
    """Numerically stable softmax along last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
