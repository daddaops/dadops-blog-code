"""Learned positional embeddings (GPT-2 / BERT style).

A lookup table of trainable position vectors, one per position.
"""
import numpy as np

# Learned positional embeddings (GPT-2 / BERT style)
max_seq_len = 1024
d_model = 64

# Random initialization — training will shape these
position_embeddings = np.random.randn(max_seq_len, d_model) * 0.02

# Look up positions and add to token embeddings
def add_learned_positions(token_embeds, pos_embeds):
    """Add learned position vectors to token embeddings."""
    seq_len = token_embeds.shape[0]
    return token_embeds + pos_embeds[:seq_len]

# Example: 5 tokens, each with a 64-dim embedding
tokens = np.random.randn(5, d_model)
positioned = add_learned_positions(tokens, position_embeddings)
print(f"Token shape: {tokens.shape}")
print(f"Positioned shape: {positioned.shape}")
