"""Load balancing loss for Mixture of Experts.

Implements the auxiliary loss from the Switch Transformer paper
that prevents router collapse by encouraging balanced expert assignment.
"""
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def top_k_route(router_probs, k=2):
    """Select top-k experts per token and renormalize weights."""
    top_k_indices = np.argpartition(-router_probs, k, axis=-1)[:, :k]
    top_k_probs = np.take_along_axis(router_probs, top_k_indices, axis=-1)
    top_k_weights = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)
    return top_k_indices, top_k_weights

def load_balance_loss(router_probs, top_k_indices, num_experts, alpha=0.01):
    """
    Auxiliary loss that encourages balanced expert assignment.

    router_probs:   (seq_len, num_experts) — full router probability distribution
    top_k_indices:  (seq_len, k) — which experts were selected
    num_experts:    int — total number of experts
    alpha:          float — loss weight (0.01 in Switch Transformer)

    Returns: scalar loss value
    """
    seq_len = router_probs.shape[0]

    # f_i: fraction of tokens where expert i is the TOP-1 choice
    primary_expert = top_k_indices[:, 0]                 # (seq_len,)
    f = np.zeros(num_experts)
    for i in range(num_experts):
        f[i] = np.sum(primary_expert == i) / seq_len     # (num_experts,)

    # P_i: mean router probability for expert i across all tokens
    P = router_probs.mean(axis=0)                        # (num_experts,)

    # Auxiliary loss: penalizes imbalance
    loss = alpha * num_experts * np.sum(f * P)            # scalar
    return loss

# Example: balanced vs. collapsed routing
np.random.seed(42)
seq_len, num_experts = 100, 8

# Balanced: each expert gets ~12.5% of tokens
balanced_probs = softmax(np.random.randn(seq_len, num_experts) * 0.5)
balanced_idx, _ = top_k_route(balanced_probs, k=2)

# Collapsed: expert 0 dominates
collapsed_logits = np.random.randn(seq_len, num_experts) * 0.5
collapsed_logits[:, 0] += 5.0  # heavily bias toward expert 0
collapsed_probs = softmax(collapsed_logits)
collapsed_idx, _ = top_k_route(collapsed_probs, k=2)

print(f"Balanced loss:  {load_balance_loss(balanced_probs, balanced_idx, num_experts):.4f}")
print(f"Collapsed loss: {load_balance_loss(collapsed_probs, collapsed_idx, num_experts):.4f}")
# Collapsed loss is much higher — the auxiliary loss detects the imbalance
