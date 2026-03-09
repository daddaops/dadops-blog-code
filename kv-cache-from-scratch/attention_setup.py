import numpy as np

np.random.seed(42)

# Tiny model: 1 layer, 1 head, embed_dim=8, head_dim=8
d_model = 8
d_head = 8

# Random projection matrices (normally these are learned)
W_q = np.random.randn(d_model, d_head) * 0.1
W_k = np.random.randn(d_model, d_head) * 0.1
W_v = np.random.randn(d_model, d_head) * 0.1

def attention(Q, K, V):
    """Standard scaled dot-product attention."""
    scores = Q @ K.T / np.sqrt(d_head)
    # Causal mask: each position can only attend to earlier positions
    seq_len = scores.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    scores = scores + mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ V
