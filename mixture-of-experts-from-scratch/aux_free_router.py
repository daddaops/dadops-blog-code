"""Auxiliary-loss-free load balancing (DeepSeek-V3 style).

Uses a bias term that adjusts routing decisions without competing
with the primary training loss via gradient descent.
"""
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# DeepSeek-V3: Bias-based load balancing (no auxiliary loss)

class AuxFreeMoERouter:
    """Router with bias-based balancing — no auxiliary loss needed."""

    def __init__(self, d_model, num_experts, gamma=0.001):
        scale = np.sqrt(2.0 / d_model)
        self.W_gate = np.random.randn(d_model, num_experts) * scale
        self.bias = np.zeros(num_experts)   # balancing bias (NOT learned by gradient)
        self.gamma = gamma                  # bias update rate

    def route(self, x, k=2):
        logits = x @ self.W_gate            # (seq_len, num_experts)

        # Routing decision uses bias (determines WHICH experts)
        routing_scores = softmax(logits + self.bias, axis=-1)
        top_k_idx = np.argpartition(-routing_scores, k, axis=-1)[:, :k]

        # Gating weights do NOT use bias (determines HOW MUCH weight)
        gating_weights = softmax(logits, axis=-1)
        top_k_probs = np.take_along_axis(gating_weights, top_k_idx, axis=-1)
        top_k_weights = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)

        # Update bias based on load (not gradient-based!)
        tokens_per_expert = np.zeros(len(self.bias))
        for idx in top_k_idx[:, 0]:
            tokens_per_expert[idx] += 1
        avg_load = tokens_per_expert.mean()
        for e in range(len(self.bias)):
            if tokens_per_expert[e] > avg_load:
                self.bias[e] -= self.gamma   # overloaded → discourage
            elif tokens_per_expert[e] < avg_load:
                self.bias[e] += self.gamma   # underloaded → encourage

        return top_k_idx, top_k_weights


if __name__ == "__main__":
    np.random.seed(42)
    d_model, num_experts = 64, 8

    router = AuxFreeMoERouter(d_model, num_experts, gamma=0.001)
    x = np.random.randn(100, d_model)

    # Run several routing rounds to see bias evolve
    for step in range(5):
        idx, weights = router.route(x, k=2)
        primary = idx[:, 0]
        counts = [np.sum(primary == e) for e in range(num_experts)]
        print(f"Step {step}: expert counts = {counts}, bias range = [{router.bias.min():.4f}, {router.bias.max():.4f}]")
