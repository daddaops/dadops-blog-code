"""
SwiGLU FFN Class

Complete SwiGLU implementation matching LLaMA's architecture.
Three weight matrices (gate, up, down), no bias terms.

Blog post: https://dadops.dev/blog/ffn-from-scratch/
"""
import numpy as np
from activations import silu

np.random.seed(42)


class SwiGLUFFN:
    """SwiGLU FFN matching LLaMA's implementation."""

    def __init__(self, d_model, d_ff):
        # Three weight matrices, no bias (modern LLMs drop biases)
        scale = np.sqrt(2.0 / d_model)
        self.w_gate = np.random.randn(d_model, d_ff) * scale  # gate projection
        self.w_up   = np.random.randn(d_model, d_ff) * scale  # up projection
        self.w_down = np.random.randn(d_ff, d_model) * scale  # down projection

    def __call__(self, x):
        gate = silu(x @ self.w_gate)     # (seq, d_ff) — gated activation
        up   = x @ self.w_up             # (seq, d_ff) — linear projection
        hidden = gate * up               # (seq, d_ff) — element-wise gating
        return hidden @ self.w_down      # (seq, d_model) — project back down

    def param_count(self):
        return sum(w.size for w in [self.w_gate, self.w_up, self.w_down])


if __name__ == "__main__":
    # Build a small version
    d_model, d_ff = 64, 171  # 8/3 * 64 ≈ 170.67, rounded up
    ffn = SwiGLUFFN(d_model, d_ff)

    x = np.random.randn(5, d_model)  # 5 tokens
    out = ffn(x)

    print(f"Input:      {x.shape}")
    print(f"Output:     {out.shape}")
    print(f"Parameters: {ffn.param_count():,}")
