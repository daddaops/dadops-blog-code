"""Complete LLaMA-style transformer block with pre-norm RMSNorm.

Multi-head attention + SwiGLU FFN. Builds a 4-block model and
shows activations remain stable across layers.
"""
import numpy as np


def rms_norm(x, gamma, eps=1e-5):
    """RMS Normalization: normalize by root-mean-square (no centering)."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def swiglu(x, W_gate, W_up, W_down):
    """SwiGLU activation: the FFN used in LLaMA."""
    gate = x @ W_gate
    up   = x @ W_up
    silu = gate * (1.0 / (1.0 + np.exp(-gate)))  # SiLU = x * sigmoid(x)
    return (silu * up) @ W_down

def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads):
    """Simplified multi-head attention."""
    B, S, D = x.shape
    head_dim = D // n_heads
    Q = (x @ Wq).reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    K = (x @ Wk).reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    V = (x @ Wv).reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    attn = softmax(scores)
    out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, S, D)
    return out @ Wo

class LLaMABlock:
    """A single LLaMA-style transformer block: Pre-Norm RMSNorm."""
    def __init__(self, d_model, n_heads, d_ff):
        scale = 0.02
        self.gamma_attn = np.ones(d_model)
        self.gamma_ffn  = np.ones(d_model)
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale
        self.W_gate = np.random.randn(d_model, d_ff) * scale
        self.W_up   = np.random.randn(d_model, d_ff) * scale
        self.W_down = np.random.randn(d_ff, d_model) * scale
        self.n_heads = n_heads

    def forward(self, x):
        # Pre-Norm attention: normalize THEN attend, add residual
        h = rms_norm(x, self.gamma_attn)
        h = multi_head_attention(h, self.Wq, self.Wk, self.Wv,
                                 self.Wo, self.n_heads)
        x = x + h   # Clean residual connection

        # Pre-Norm FFN: normalize THEN transform, add residual
        h = rms_norm(x, self.gamma_ffn)
        h = swiglu(h, self.W_gate, self.W_up, self.W_down)
        x = x + h   # Clean residual connection

        return x


if __name__ == "__main__":
    # Build a 4-block model and process a sequence
    np.random.seed(42)
    d_model, n_heads, d_ff = 64, 4, 172  # ~2.7x expansion (LLaMA ratio)

    blocks = [LLaMABlock(d_model, n_heads, d_ff) for _ in range(4)]

    x = np.random.randn(1, 8, d_model)   # batch=1, seq_len=8, dim=64
    print(f"Input:    shape = {x.shape}, std = {x.std():.4f}")

    for i, block in enumerate(blocks):
        x = block.forward(x)
        print(f"Block {i+1}:  shape = {x.shape}, std = {x.std():.4f}, "
              f"mean = {x.mean():.4f}, max|x| = {np.abs(x).max():.4f}")
