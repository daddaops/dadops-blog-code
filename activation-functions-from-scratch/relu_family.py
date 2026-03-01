"""
ReLU and its variants: Leaky ReLU, Parametric ReLU, ELU, SELU.

Shows how each variant addresses the "dying neuron" problem
by keeping a non-zero gradient for negative inputs.

Requires: numpy

From: https://dadops.dev/blog/activation-functions-from-scratch/
"""

import numpy as np

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Small slope for negative inputs prevents dying neurons."""
    return np.where(x > 0, x, alpha * x)

def parametric_relu(x, alpha):
    """Alpha is learned during training, not fixed."""
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """Exponential curve for negatives: smooth at zero, saturates at -alpha."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x, alpha=1.6733, lam=1.0507):
    """Self-normalizing: preserves mean ~0 and variance ~1 through layers."""
    return lam * np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Compare on negative input: which ones keep the gradient alive?
x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

print("x:          ", x)
print("ReLU:       ", np.round(relu(x), 4))
print("Leaky ReLU: ", np.round(leaky_relu(x), 4))
print("ELU:        ", np.round(elu(x), 4))
print("SELU:       ", np.round(selu(x), 4))

# At x=-2:  ReLU=0, Leaky=−0.02, ELU=−0.865, SELU=−1.520
# ReLU kills the signal; the variants keep it alive
