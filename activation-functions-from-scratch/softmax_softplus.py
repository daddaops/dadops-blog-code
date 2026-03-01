"""
Specialized activations: Softmax and Softplus.

Softmax turns arbitrary scores into a probability distribution.
Softplus is a smooth approximation to ReLU: log(1 + exp(x)).

Requires: numpy

From: https://dadops.dev/blog/activation-functions-from-scratch/
"""

import numpy as np

def softmax(x):
    """Softmax with numerical stability trick (subtract max)."""
    # Without subtracting max, exp(1000) overflows to inf
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softplus(x):
    """Smooth approximation to ReLU: log(1 + exp(x))."""
    # Numerically stable version
    return np.where(x > 20, x, np.log(1 + np.exp(x)))

# Softmax: turns arbitrary scores into probabilities
logits = np.array([2.0, 1.0, 0.5, -1.0])
probs = softmax(logits)
print(f"Logits:       {logits}")
print(f"Probabilities: {np.round(probs, 4)}")
print(f"Sum:          {probs.sum():.6f}")  # 1.000000

# Softplus vs ReLU: smooth vs kinked
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"\nx:        {x}")
print(f"ReLU:     {np.maximum(0, x)}")
print(f"Softplus: {np.round(softplus(x), 4)}")
# At x=0: ReLU=0, Softplus=0.6931 (smooth transition)
