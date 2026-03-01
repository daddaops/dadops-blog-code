"""
Smooth activation functions: GELU, SiLU/Swish, and Mish.

These are the modern activations used in transformers (GPT, BERT, ViT).
All three converge on a similar shape: pass positive values, kill negatives,
with a smooth transition zone.

Requires: numpy, scipy

From: https://dadops.dev/blog/activation-functions-from-scratch/
"""

import numpy as np
from scipy.special import erf

def gelu_exact(x):
    """GELU using the Gaussian CDF: x * Phi(x)."""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_approx(x):
    """Fast tanh approximation used in practice (e.g., PyTorch)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def silu(x):
    """SiLU / Swish: x * sigmoid(x). Nearly identical to GELU."""
    return x * (1 / (1 + np.exp(-x)))

def mish(x):
    """Mish: x * tanh(softplus(x)). Slightly smoother than SiLU."""
    return x * np.tanh(np.log(1 + np.exp(x)))

x = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
print("x:          ", x)
print("GELU exact: ", np.round(gelu_exact(x), 4))
print("GELU approx:", np.round(gelu_approx(x), 4))
print("SiLU/Swish: ", np.round(silu(x), 4))
print("Mish:       ", np.round(mish(x), 4))

# At x=-0.5: GELU=-0.1543, SiLU=-0.1888, Mish=-0.1894
# At x= 3.0: all three ≈ 2.996 (converge for large positive)
