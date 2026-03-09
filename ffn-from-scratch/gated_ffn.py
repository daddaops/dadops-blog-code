"""
Gated Linear Unit (GLU) FFN Variants

Implements GLU, ReGLU, GEGLU, and SwiGLU — the gated architectures
used in modern LLMs like LLaMA and PaLM.

Blog post: https://dadops.dev/blog/ffn-from-scratch/
"""
import numpy as np
from activations import gelu_exact, silu

np.random.seed(42)


def glu_ffn(x, W_gate, W_up, W_down, activation='swiglu'):
    """Gated Linear Unit FFN with selectable activation."""
    gate_proj = x @ W_gate        # (seq, d_model) @ (d_model, d_ff) = (seq, d_ff)
    up_proj   = x @ W_up          # (seq, d_model) @ (d_model, d_ff) = (seq, d_ff)

    # Apply activation to the gate branch
    if activation == 'glu':
        gate = 1 / (1 + np.exp(-gate_proj))                # sigmoid (gate only)
    elif activation == 'reglu':
        gate = np.maximum(0, gate_proj)                     # ReLU
    elif activation == 'geglu':
        gate = gelu_exact(gate_proj)                        # GELU
    elif activation == 'swiglu':
        gate = silu(gate_proj)                              # SiLU/Swish
    else:
        raise ValueError(f"Unknown activation: {activation}")

    hidden = gate * up_proj       # element-wise gating: (seq, d_ff)
    output = hidden @ W_down      # (seq, d_ff) @ (d_ff, d_model) = (seq, d_model)
    return output


# Build it with small dimensions
d_model, d_ff = 8, 22  # ~8/3 * 8 = ~21.3, rounded to 22
W_gate = np.random.randn(d_model, d_ff) * 0.1
W_up   = np.random.randn(d_model, d_ff) * 0.1
W_down = np.random.randn(d_ff, d_model) * 0.1

x = np.random.randn(3, d_model)

for variant in ['glu', 'reglu', 'geglu', 'swiglu']:
    out = glu_ffn(x, W_gate, W_up, W_down, activation=variant)
    print(f"{variant:6s} | output[0,:4] = {out[0, :4].round(4)}")
