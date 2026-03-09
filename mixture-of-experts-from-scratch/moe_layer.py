"""Complete Mixture of Experts layer with SwiGLU experts.

Implements the SwiGLUExpert and MoELayer classes, then compares
parameter counts between dense FFN and MoE.
"""
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def silu(x):
    """SiLU/Swish activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x)))

class SwiGLUExpert:
    """A single SwiGLU FFN expert (from the FFN post)."""

    def __init__(self, d_model, d_ff):
        scale = np.sqrt(2.0 / d_model)
        self.w_gate = np.random.randn(d_model, d_ff) * scale
        self.w_up   = np.random.randn(d_model, d_ff) * scale
        self.w_down = np.random.randn(d_ff, d_model) * scale

    def __call__(self, x):
        gate = silu(x @ self.w_gate)  # (tokens, d_ff)
        up   = x @ self.w_up          # (tokens, d_ff)
        return (gate * up) @ self.w_down  # (tokens, d_model)

    def param_count(self):
        return sum(w.size for w in [self.w_gate, self.w_up, self.w_down])

class MoELayer:
    """Mixture of Experts layer that replaces a dense FFN."""

    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, alpha=0.01):
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.aux_loss = 0.0  # stored after each forward pass

        # N independent SwiGLU experts
        self.experts = [SwiGLUExpert(d_model, d_ff) for _ in range(num_experts)]

        # Learned router: d_model -> num_experts
        scale = np.sqrt(2.0 / d_model)
        self.W_gate = np.random.randn(d_model, num_experts) * scale

    def __call__(self, x):
        """
        x: (seq_len, d_model) — input from attention + residual + norm
        Returns: (seq_len, d_model) — MoE output
        """
        seq_len, d_model = x.shape

        # Step 1: Router produces probability distribution over experts
        logits = x @ self.W_gate                        # (seq_len, num_experts)
        router_probs = softmax(logits, axis=-1)         # (seq_len, num_experts)

        # Step 2: Top-k expert selection per token
        top_k_idx = np.argpartition(
            -router_probs, self.top_k, axis=-1
        )[:, :self.top_k]                               # (seq_len, top_k)
        top_k_probs = np.take_along_axis(
            router_probs, top_k_idx, axis=-1
        )                                               # (seq_len, top_k)
        top_k_weights = top_k_probs / top_k_probs.sum(
            axis=-1, keepdims=True
        )                                               # (seq_len, top_k) — renormalized

        # Step 3: Dispatch tokens to experts and combine outputs
        output = np.zeros_like(x)                       # (seq_len, d_model)
        for k in range(self.top_k):
            expert_indices = top_k_idx[:, k]            # (seq_len,) — which expert
            weights = top_k_weights[:, k]               # (seq_len,) — how much weight

            for e in range(self.num_experts):
                # Find tokens assigned to this expert at rank k
                mask = (expert_indices == e)
                if not np.any(mask):
                    continue

                tokens_for_expert = x[mask]             # (n_tokens, d_model)
                expert_out = self.experts[e](tokens_for_expert)  # (n_tokens, d_model)
                output[mask] += weights[mask, None] * expert_out  # weighted contribution

        # Step 4: Compute auxiliary load-balancing loss
        primary = top_k_idx[:, 0]
        f = np.array([np.sum(primary == e) / seq_len
                       for e in range(self.num_experts)])
        P = router_probs.mean(axis=0)
        self.aux_loss = self.alpha * self.num_experts * np.sum(f * P)

        return output                                   # (seq_len, d_model)

    def param_count(self):
        expert_params = sum(e.param_count() for e in self.experts)
        router_params = self.W_gate.size
        return expert_params + router_params


if __name__ == "__main__":
    np.random.seed(42)
    d_model = 64
    d_ff = 172  # same as our transformer capstone

    # Dense FFN: one big SwiGLU
    dense_ffn = SwiGLUExpert(d_model, d_ff)

    # MoE: 8 experts, each with the SAME d_ff, top-2 routing
    moe = MoELayer(d_model, d_ff, num_experts=8, top_k=2)

    x = np.random.randn(10, d_model)  # 10 tokens

    dense_out = dense_ffn(x)
    moe_out = moe(x)

    print(f"Dense FFN:  {dense_out.shape}  |  params: {dense_ffn.param_count():>8,}")
    print(f"MoE Layer:  {moe_out.shape}  |  params: {moe.param_count():>8,}")
    print(f"Aux loss:   {moe.aux_loss:.4f}")
    print(f"Parameter ratio: {moe.param_count() / dense_ffn.param_count():.1f}x")
