"""Pre-activation (v2) vs post-activation (v1) residual blocks."""
import numpy as np
from helpers import conv2d_same, batch_norm

def post_activation_block(x, W1, W2, gamma1, beta1, gamma2, beta2):
    """Original ResNet v1: BN-ReLU after each conv, ReLU after addition."""
    out = conv2d_same(x, W1)
    out = batch_norm(out, gamma1, beta1)
    out = np.maximum(0, out)        # ReLU
    out = conv2d_same(out, W2)
    out = batch_norm(out, gamma2, beta2)
    out = out + x                   # Skip connection
    out = np.maximum(0, out)        # ReLU gates the identity!
    return out

def pre_activation_block(x, W1, W2, gamma1, beta1, gamma2, beta2):
    """ResNet v2: BN-ReLU before each conv, clean identity path."""
    out = batch_norm(x, gamma1, beta1)
    out = np.maximum(0, out)        # ReLU
    out = conv2d_same(out, W1)
    out = batch_norm(out, gamma2, beta2)
    out = np.maximum(0, out)        # ReLU
    out = conv2d_same(out, W2)
    out = out + x                   # Pure identity — no ReLU gating!
    return out

# The identity path in v2 is: x --> (+) --> next block
# Nothing modifies x along the skip connection.
# This is why transformers use Pre-Norm: x = x + Sublayer(Norm(x))
# Same idea, discovered independently for attention layers.

# Smoke test
rng = np.random.RandomState(42)
channels = 4
scale = np.sqrt(2.0 / (channels * 9))
W1 = rng.randn(channels, channels, 3, 3) * scale
W2 = rng.randn(channels, channels, 3, 3) * scale
gamma1, beta1 = np.ones(channels), np.zeros(channels)
gamma2, beta2 = np.ones(channels), np.zeros(channels)

x = rng.randn(channels, 8, 8)
out_v1 = post_activation_block(x, W1, W2, gamma1, beta1, gamma2, beta2)
out_v2 = pre_activation_block(x, W1, W2, gamma1, beta1, gamma2, beta2)
print(f"Post-activation (v1) output shape: {out_v1.shape}")
print(f"Pre-activation  (v2) output shape: {out_v2.shape}")
print(f"v1 has negative values: {(out_v1 < 0).any()}")  # False — ReLU after add
print(f"v2 has negative values: {(out_v2 < 0).any()}")  # True — no ReLU after add
