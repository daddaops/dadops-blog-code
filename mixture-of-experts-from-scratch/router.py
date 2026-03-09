"""Router and top-k routing for Mixture of Experts.

Implements the gating network that routes tokens to experts,
and top-k selection with weight renormalization.
"""
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class Router:
    """Learned gating network that routes tokens to experts."""

    def __init__(self, d_model, num_experts):
        # One linear layer: d_model -> num_experts
        scale = np.sqrt(2.0 / d_model)
        self.W_gate = np.random.randn(d_model, num_experts) * scale

    def __call__(self, x):
        """
        x: (seq_len, d_model) — token hidden states
        Returns: (seq_len, num_experts) — probability over experts per token
        """
        logits = x @ self.W_gate          # (seq_len, num_experts)
        return softmax(logits, axis=-1)   # (seq_len, num_experts)

# --- Demo: Router output ---
np.random.seed(42)
router = Router(d_model=64, num_experts=8)
x = np.random.randn(4, 64)  # 4 tokens

probs = router(x)
print("Router output — probability per expert for each token:")
for i in range(4):
    top2 = np.argsort(probs[i])[-2:][::-1]
    print(f"  Token {i}: ", end="")
    for j in range(8):
        marker = " *" if j in top2 else "  "
        print(f"E{j}={probs[i,j]:.3f}{marker}", end=" ")
    print()
# Each row sums to 1.0 — it's a valid probability distribution

print()

# --- Top-k routing ---
def top_k_route(router_probs, k=2):
    """
    Select top-k experts per token and renormalize weights.

    router_probs: (seq_len, num_experts) — probability distribution
    k: number of experts to select per token

    Returns:
        top_k_indices: (seq_len, k) — which experts are selected
        top_k_weights: (seq_len, k) — renormalized weights
    """
    seq_len, num_experts = router_probs.shape

    # Find the k largest probabilities per token
    top_k_indices = np.argpartition(-router_probs, k, axis=-1)[:, :k]  # (seq_len, k)

    # Gather the corresponding probabilities
    top_k_probs = np.take_along_axis(router_probs, top_k_indices, axis=-1)  # (seq_len, k)

    # Renormalize so the k weights sum to 1
    top_k_weights = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)  # (seq_len, k)

    return top_k_indices, top_k_weights

# --- Demo: Top-k routing ---
np.random.seed(7)
router = Router(d_model=8, num_experts=4)
x = np.random.randn(4, 8)

probs = router(x)
indices, weights = top_k_route(probs, k=2)

print("Token | Router Probs (all 4 experts)       | Selected → Weights")
print("-" * 75)
for i in range(4):
    prob_str = " ".join(f"E{j}={probs[i,j]:.3f}" for j in range(4))
    sel_str = " + ".join(f"E{indices[i,j]}×{weights[i,j]:.3f}" for j in range(2))
    print(f"  {i}   | {prob_str} | {sel_str}")

# Output: each token picks its top 2 experts, weights sum to 1.0
