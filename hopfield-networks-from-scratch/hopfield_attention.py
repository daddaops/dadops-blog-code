import numpy as np
from hopfield_continuous import softmax, continuous_hopfield_update

def attention(query, keys, values, d_k):
    """Standard scaled dot-product attention for a single query."""
    scores = np.array([query @ k / np.sqrt(d_k) for k in keys])
    weights = softmax(scores)
    return sum(w * v for w, v in zip(weights, values))

if __name__ == "__main__":
    # Setup: 4 stored patterns in R^8
    np.random.seed(123)
    d = 8
    patterns = [np.random.randn(d) for _ in range(4)]
    query = np.random.randn(d)

    # Hopfield update with beta = 1/sqrt(d)
    beta = 1.0 / np.sqrt(d)
    hopfield_out = continuous_hopfield_update(patterns, query, beta=beta)

    # Attention with same data: keys = values = patterns
    attn_out = attention(query, keys=patterns, values=patterns, d_k=d)

    # Compare
    print(f"Hopfield output: [{', '.join(f'{v:.4f}' for v in hopfield_out[:4])}...]")
    print(f"Attention output: [{', '.join(f'{v:.4f}' for v in attn_out[:4])}...]")
    print(f"Max difference:   {np.max(np.abs(hopfield_out - attn_out)):.2e}")
    # Hopfield output: [-0.8643, 0.5116, 0.4454, -1.3278...]
    # Attention output: [-0.8643, 0.5116, 0.4454, -1.3278...]
    # Max difference:   2.22e-16
