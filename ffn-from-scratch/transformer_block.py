"""
Complete Transformer Block

Combines RMSNorm, single-head attention, and SwiGLU FFN in
Pre-Norm configuration. Shows FFN accounts for ~66% of parameters.

Blog post: https://dadops.dev/blog/ffn-from-scratch/
"""
import numpy as np
from activations import silu
from swiglu_class import SwiGLUFFN

np.random.seed(42)


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: normalize by root-mean-square, then scale."""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def simplified_attention(x, Wq, Wk, Wv, Wo):
    """Single-head attention (simplified for clarity)."""
    Q = x @ Wq  # (seq, d)
    K = x @ Wk  # (seq, d)
    V = x @ Wv  # (seq, d)
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)    # (seq, seq)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)  # softmax
    return (weights @ V) @ Wo             # (seq, d)


class TransformerBlock:
    """One complete transformer block: attention + FFN."""

    def __init__(self, d_model, d_ff):
        s = np.sqrt(2.0 / d_model)
        # Attention weights (single-head for simplicity)
        self.Wq = np.random.randn(d_model, d_model) * s
        self.Wk = np.random.randn(d_model, d_model) * s
        self.Wv = np.random.randn(d_model, d_model) * s
        self.Wo = np.random.randn(d_model, d_model) * s
        # FFN weights (SwiGLU)
        self.ffn = SwiGLUFFN(d_model, d_ff)
        # Norm weights (all ones = identity scaling)
        self.norm1 = np.ones(d_model)
        self.norm2 = np.ones(d_model)

    def __call__(self, x):
        # Pre-Norm: normalize BEFORE each sublayer
        # Sublayer 1: attention (mixes tokens)
        x = x + simplified_attention(
            rms_norm(x, self.norm1),
            self.Wq, self.Wk, self.Wv, self.Wo
        )
        # Sublayer 2: FFN (processes each token independently)
        x = x + self.ffn(rms_norm(x, self.norm2))
        return x

    def param_count(self):
        attn_params = 4 * self.Wq.size           # Q, K, V, O projections
        ffn_params  = self.ffn.param_count()       # gate, up, down projections
        norm_params = self.norm1.size + self.norm2.size
        return attn_params, ffn_params, norm_params


# Build a block
d_model, d_ff = 64, 171
block = TransformerBlock(d_model, d_ff)

x = np.random.randn(10, d_model)  # 10 tokens
out = block(x)

attn_p, ffn_p, norm_p = block.param_count()
total = attn_p + ffn_p + norm_p
print(f"Attention params: {attn_p:>8,}  ({attn_p/total*100:.1f}%)")
print(f"FFN params:       {ffn_p:>8,}  ({ffn_p/total*100:.1f}%)")
print(f"Norm params:      {norm_p:>8,}  ({norm_p/total*100:.1f}%)")
print(f"Total:            {total:>8,}")
