"""Training comparison: MoE vs Dense FFN.

Simplified forward-only comparison on a character-level next-token
prediction task, tracking loss over 50 steps.
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
    """A single SwiGLU FFN expert."""
    def __init__(self, d_model, d_ff):
        scale = np.sqrt(2.0 / d_model)
        self.w_gate = np.random.randn(d_model, d_ff) * scale
        self.w_up   = np.random.randn(d_model, d_ff) * scale
        self.w_down = np.random.randn(d_ff, d_model) * scale

    def __call__(self, x):
        gate = silu(x @ self.w_gate)
        up   = x @ self.w_up
        return (gate * up) @ self.w_down

    def param_count(self):
        return sum(w.size for w in [self.w_gate, self.w_up, self.w_down])

class MoELayer:
    """Mixture of Experts layer that replaces a dense FFN."""
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, alpha=0.01):
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.aux_loss = 0.0
        self.experts = [SwiGLUExpert(d_model, d_ff) for _ in range(num_experts)]
        scale = np.sqrt(2.0 / d_model)
        self.W_gate = np.random.randn(d_model, num_experts) * scale

    def __call__(self, x):
        seq_len, d_model = x.shape
        logits = x @ self.W_gate
        router_probs = softmax(logits, axis=-1)
        top_k_idx = np.argpartition(-router_probs, self.top_k, axis=-1)[:, :self.top_k]
        top_k_probs = np.take_along_axis(router_probs, top_k_idx, axis=-1)
        top_k_weights = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)

        output = np.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = top_k_idx[:, k]
            weights = top_k_weights[:, k]
            for e in range(self.num_experts):
                mask = (expert_indices == e)
                if not np.any(mask):
                    continue
                tokens_for_expert = x[mask]
                expert_out = self.experts[e](tokens_for_expert)
                output[mask] += weights[mask, None] * expert_out

        primary = top_k_idx[:, 0]
        f = np.array([np.sum(primary == e) / seq_len for e in range(self.num_experts)])
        P = router_probs.mean(axis=0)
        self.aux_loss = self.alpha * self.num_experts * np.sum(f * P)
        return output

# Simplified training comparison: MoE vs Dense FFN
# (Single-layer, character-level, SGD for clarity)

np.random.seed(42)
d_model, d_ff, vocab_size = 32, 86, 26  # small for CPU training
seq_len, lr = 16, 0.01

# Simple embedding + output head (shared between both models)
embed = np.random.randn(vocab_size, d_model) * 0.02

# Two models: dense FFN vs MoE (4 experts, top-2)
dense = SwiGLUExpert(d_model, d_ff)
moe = MoELayer(d_model, d_ff, num_experts=4, top_k=2, alpha=0.01)

# Toy training data: repeating character patterns
text = "abcdefghijklmnopqrstuvwxyz" * 20  # 520 characters
data = np.array([ord(c) - ord('a') for c in text])  # 0-25

def get_batch(data, seq_len, batch_start):
    x = data[batch_start:batch_start + seq_len]
    y = data[batch_start + 1:batch_start + seq_len + 1]
    return x, y

def forward_and_loss(model, embed, x_ids, y_ids):
    """Forward pass through embedding -> model -> softmax -> loss."""
    x = embed[x_ids]                         # (seq_len, d_model)
    h = model(x)                             # (seq_len, d_model)
    logits = h @ embed.T                     # (seq_len, vocab_size)

    # Cross-entropy loss
    logits_max = logits.max(axis=-1, keepdims=True)
    log_probs = logits - logits_max - np.log(
        np.exp(logits - logits_max).sum(axis=-1, keepdims=True)
    )
    loss = -log_probs[np.arange(len(y_ids)), y_ids].mean()
    return loss

# Training loop — track loss over 50 steps
dense_losses, moe_losses = [], []
for step in range(50):
    batch_start = (step * seq_len) % (len(data) - seq_len - 1)
    x_ids, y_ids = get_batch(data, seq_len, batch_start)

    d_loss = forward_and_loss(dense, embed, x_ids, y_ids)
    m_loss = forward_and_loss(moe, embed, x_ids, y_ids)

    dense_losses.append(float(d_loss))
    moe_losses.append(float(m_loss))

    if step % 10 == 0:
        aux = moe.aux_loss
        print(f"Step {step:3d} | Dense loss: {d_loss:.3f} | "
              f"MoE loss: {m_loss:.3f} (aux: {aux:.4f})")
