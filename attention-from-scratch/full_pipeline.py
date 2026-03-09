"""Full Attention Pipeline: Embeddings + Positional Encoding + Multi-Head Attention.

Wires up the complete pipeline end-to-end on a toy sentence.
"""
import numpy as np
from multi_head_attention import MultiHeadAttention
from positional_encoding import positional_encoding

if __name__ == "__main__":
    np.random.seed(42)

    # Configuration
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(sentence)
    d_model = 64
    num_heads = 8

    # Simulate token embeddings (random, since we don't have a vocabulary)
    token_embeddings = np.random.randn(n, d_model) * 0.5

    # Add positional encoding
    pe = positional_encoding(n, d_model)
    X = token_embeddings + pe

    print("Before attention:")
    print(f"  Embedding norms: {[f'{np.linalg.norm(X[i]):.2f}' for i in range(n)]}")

    # Run multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    output, weights = mha.forward(X)

    print("\nAfter attention:")
    print(f"  Output norms: {[f'{np.linalg.norm(output[i]):.2f}' for i in range(n)]}")
    print(f"\nEvery token now carries information from every other token.")
    print(f"Shape in: {X.shape} → Shape out: {output.shape}")
