"""Round-to-Nearest quantization on network outputs."""
import numpy as np
from helpers import symmetric_quantize

def simple_linear(x, W, b):
    """A plain linear layer: y = xW^T + b."""
    return x @ W.T + b

def quantized_linear(x, W, b, bits=8):
    """Linear layer with weight quantization."""
    q, scale, W_deq = symmetric_quantize(W.flatten(), bits)
    W_deq = W_deq.reshape(W.shape)
    return x @ W_deq.T + b

# Build a tiny trained network (2 layers, classification)
np.random.seed(0)
W1 = np.random.randn(32, 16).astype(np.float32) * 0.3
b1 = np.zeros(32)
W2 = np.random.randn(4, 32).astype(np.float32) * 0.3
b2 = np.zeros(4)

# Generate some test data
X_test = np.random.randn(200, 16).astype(np.float32)

def forward(x, bits=32):
    """Forward pass — quantize weights if bits < 32."""
    if bits < 32:
        _, _, W1_q = symmetric_quantize(W1.flatten(), bits)
        _, _, W2_q = symmetric_quantize(W2.flatten(), bits)
        h = np.maximum(0, x @ W1_q.reshape(W1.shape).T + b1)
        return h @ W2_q.reshape(W2.shape).T + b2
    else:
        h = np.maximum(0, x @ W1.T + b1)
        return h @ W2.T + b2

# Compare outputs at different bit widths
ref_output = forward(X_test, bits=32)
for bits in [8, 4, 3, 2]:
    q_output = forward(X_test, bits=bits)
    mse = np.mean((ref_output - q_output) ** 2)
    max_err = np.max(np.abs(ref_output - q_output))
    print(f"INT{bits}: Output MSE={mse:.6f}, Max deviation={max_err:.4f}")
