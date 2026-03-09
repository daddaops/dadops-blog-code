"""RMS Normalization: normalize by root-mean-square (no centering).

Compares statistics across BatchNorm, LayerNorm, and RMSNorm.
"""
import numpy as np


def batch_norm(x, gamma, beta, eps=1e-5):
    """Batch Normalization: normalize across the batch dimension."""
    mean = x.mean(axis=0, keepdims=True)
    var  = x.var(axis=0, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer Normalization: normalize across the feature dimension."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def rms_norm(x, gamma, eps=1e-5):
    """RMS Normalization: normalize by root-mean-square (no centering)."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma    # No beta parameter

# Compare all three on the same input
np.random.seed(42)
x = np.random.randn(4, 128) * 3 + 1   # Shifted, scaled

gamma = np.ones(128)
beta  = np.zeros(128)

bn_out  = batch_norm(x, gamma, beta)
ln_out  = layer_norm(x, gamma, beta)
rms_out = rms_norm(x, gamma)

print("Output statistics (first sample):")
print(f"  BatchNorm:  mean = {bn_out[0].mean():+.4f}, std = {bn_out[0].std():.4f}")
print(f"  LayerNorm:  mean = {ln_out[0].mean():+.4f}, std = {ln_out[0].std():.4f}")
print(f"  RMSNorm:    mean = {rms_out[0].mean():+.4f}, std = {rms_out[0].std():.4f}")
