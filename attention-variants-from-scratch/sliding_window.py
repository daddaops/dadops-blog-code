"""Sliding Window Attention — Local Is Enough.

Instead of attending to the entire sequence, each token only sees the
previous w tokens. Information propagates across layers for long-range deps.
"""
import numpy as np
np.random.seed(42)

def create_attention_mask(seq_len, window_size=None):
    """Create causal attention mask, optionally with sliding window."""
    # Causal mask: token i can attend to tokens 0..i
    mask = np.tril(np.ones((seq_len, seq_len)))

    if window_size is not None:
        # Sliding window: token i can only attend to tokens (i-w)..i
        window_mask = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            window_mask[i, start:i+1] = 1.0
        mask = mask * window_mask

    return mask

def sliding_window_attention(x, d_head, window_size):
    """Attention with a sliding window mask."""
    seq_len, d_model = x.shape
    W_Q = np.random.randn(d_model, d_head) * 0.1
    W_K = np.random.randn(d_model, d_head) * 0.1
    W_V = np.random.randn(d_model, d_head) * 0.1

    Q, K, V = x @ W_Q, x @ W_K, x @ W_V
    scores = Q @ K.T / np.sqrt(d_head)

    mask = create_attention_mask(seq_len, window_size)
    scores = np.where(mask == 1, scores, -1e9)

    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights

if __name__ == "__main__":
    # Show how window size affects the attention pattern
    seq_len = 12
    x = np.random.randn(seq_len, 32)

    print("Attention masks (1 = can attend, 0 = masked):\n")
    for w_name, w in [("Full", None), ("Window=4", 4), ("Window=2", 2)]:
        mask = create_attention_mask(seq_len, w)
        # Show cache size: full stores all tokens, window stores only w
        cache = seq_len if w is None else min(seq_len, w)
        print(f"{w_name} (max cache = {cache} tokens per layer):")
        for row in mask[:6, :6]:
            print("  ", " ".join("\u2588" if v else "\u00b7" for v in row))
        print()
