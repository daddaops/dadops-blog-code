"""Attention is permutation-invariant without positional encoding.

Shows that 'dog bites man' and 'man bites dog' produce
the same attention weights (just permuted rows/columns).
"""
import numpy as np
np.random.seed(42)

# Random embeddings for three words
embeddings = {
    'dog': np.random.randn(8),
    'bites': np.random.randn(8),
    'man': np.random.randn(8),
}

def attention_weights(tokens, embeds):
    """Compute attention weight matrix for a token sequence."""
    X = np.stack([embeds[t] for t in tokens])  # (3, 8)
    scores = X @ X.T / np.sqrt(8)              # (3, 3)
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)

w1 = attention_weights(['dog', 'bites', 'man'], embeddings)
w2 = attention_weights(['man', 'bites', 'dog'], embeddings)

print("'dog bites man' weights:")
print(np.round(w1, 3))

print("\n'man bites dog' weights:")
print(np.round(w2, 3))
