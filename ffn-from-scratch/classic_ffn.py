"""
Classic Transformer FFN

The original two-matrix feed-forward network: ReLU(xW1 + b1)W2 + b2.
Expand to d_ff, activate, project back to d_model.

Blog post: https://dadops.dev/blog/ffn-from-scratch/
"""
import numpy as np
np.random.seed(42)

def classic_ffn(x, W1, b1, W2, b2):
    """Original transformer FFN: ReLU(xW1 + b1)W2 + b2"""
    hidden = x @ W1 + b1          # (seq, d_model) @ (d_model, d_ff) = (seq, d_ff)
    activated = np.maximum(0, hidden)  # ReLU: zero out negatives
    output = activated @ W2 + b2  # (seq, d_ff) @ (d_ff, d_model) = (seq, d_model)
    return output

# Dimensions from the original transformer paper
d_model, d_ff = 8, 32  # using small dims for demonstration (real: 512, 2048)
seq_len = 3

# Initialize weights (Kaiming/He initialization for ReLU)
W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
b2 = np.zeros(d_model)

# Three tokens of input
x = np.random.randn(seq_len, d_model)  # (3, 8)

output = classic_ffn(x, W1, b1, W2, b2)
print(f"Input shape:  {x.shape}")       # (3, 8)
print(f"Output shape: {output.shape}")   # (3, 8)
print(f"Parameters:   {d_model * d_ff + d_ff + d_ff * d_model + d_model}")  # 8*32 + 32 + 32*8 + 8 = 552
print(f"First token in:  {x[0, :4].round(3)}")
print(f"First token out: {output[0, :4].round(3)}")
