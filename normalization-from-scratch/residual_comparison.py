"""Residual connection comparison: no-norm vs post-norm vs pre-norm.

Shows that pre-norm keeps activations stable across 50 layers
while no-norm explodes and post-norm is perfectly normalized.
"""
import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer Normalization: normalize across the feature dimension."""
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta


np.random.seed(42)
d = 128
gamma_ln = np.ones(d)
beta_ln  = np.zeros(d)

def residual_postnorm(x, weights):
    """Post-Norm: x = LayerNorm(x + sublayer(x))"""
    stds = []
    for W in weights:
        sublayer = x @ W
        x = layer_norm(x + sublayer, gamma_ln, beta_ln)
        stds.append(x.std())
    return stds

def residual_prenorm(x, weights):
    """Pre-Norm: x = x + sublayer(LayerNorm(x))"""
    stds = []
    for W in weights:
        sublayer = layer_norm(x, gamma_ln, beta_ln) @ W
        x = x + sublayer
        stds.append(x.std())
    return stds

def residual_nonorm(x, weights):
    """No normalization baseline"""
    stds = []
    for W in weights:
        x = x + x @ W
        stds.append(x.std())
    return stds

weights = [np.random.randn(d, d) * 0.05 for _ in range(50)]
x = np.random.randn(4, d)

none_stds = residual_nonorm(x.copy(), weights)
post_stds = residual_postnorm(x.copy(), weights)
pre_stds  = residual_prenorm(x.copy(), weights)

print("Layer  | No Norm        | Post-Norm      | Pre-Norm")
print("-------|----------------|----------------|----------")
for i in [0, 9, 24, 49]:
    print(f"  {i+1:3d}   | {none_stds[i]:14.4e} | {post_stds[i]:14.4f} | {pre_stds[i]:.4f}")
