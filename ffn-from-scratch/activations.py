"""
Activation Function Comparison

Compares ReLU, GELU, and SiLU (Swish) activations and their gradients.
Shows how smooth activations avoid the dead neuron problem.

Blog post: https://dadops.dev/blog/ffn-from-scratch/
"""
import numpy as np
from scipy.special import erf  # for exact GELU


def relu(x):
    return np.maximum(0, x)

def gelu_exact(x):
    """GELU: x * Phi(x) where Phi is the standard normal CDF"""
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

def gelu_approx(x):
    """Fast tanh approximation used in BERT/GPT-2"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def silu(x):
    """SiLU/Swish: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x)))


def relu_grad(x):
    """Gradient of ReLU: 1 if x > 0, else 0"""
    return 1.0 if x > 0 else 0.0

def gelu_grad(x):
    """Gradient of GELU: Phi(x) + x * phi(x)"""
    phi = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)  # standard normal PDF
    Phi = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))      # standard normal CDF
    return Phi + x * phi

def silu_grad(x):
    """Gradient of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))"""
    sig = 1.0 / (1.0 + np.exp(-x))
    return sig + x * sig * (1.0 - sig)


if __name__ == "__main__":
    test_points = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    print("     x   |  ReLU  |  GELU  |  SiLU")
    print("---------+--------+--------+-------")
    for xi in test_points:
        print(f"  {xi:5.1f}  | {relu(xi):6.3f} | {gelu_exact(xi):6.3f} | {silu(xi):6.3f}")

    print()

    print("     x   | ReLU' | GELU'  | SiLU'")
    print("---------+-------+--------+------")
    for xi in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        print(f"  {xi:5.1f}  | {relu_grad(xi):5.3f} | {gelu_grad(xi):6.3f} | {silu_grad(xi):6.3f}")
