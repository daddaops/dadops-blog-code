import numpy as np

def maxsim_score(q_embeddings, d_embeddings):
    """
    ColBERT MaxSim scoring.
    q_embeddings: (n_q, dim) -- one embedding per query token
    d_embeddings: (n_d, dim) -- one embedding per document token

    For each query token, find the max similarity to any doc token.
    Sum these max similarities across all query tokens.
    """
    # Token-token similarity matrix: (n_q, n_d)
    sim_matrix = q_embeddings @ d_embeddings.T

    # MaxSim: max over document tokens for each query token
    max_sims = sim_matrix.max(axis=1)   # (n_q,)
    return float(max_sims.sum())

# Compare single-vector vs MaxSim
rng = np.random.RandomState(42)

# Simulate encoding "fix leaking faucet" (3 tokens)
q_tokens = rng.randn(3, 64)
q_tokens = q_tokens / np.linalg.norm(q_tokens, axis=1, keepdims=True)

# Document 1: "plumbing repair stopping drips fixtures" (5 tokens)
# -- semantically similar but different words
d1_tokens = q_tokens[0:1] * 0.7 + rng.randn(5, 64) * 0.3
d1_tokens = d1_tokens / np.linalg.norm(d1_tokens, axis=1, keepdims=True)

# Document 2: random unrelated document (5 tokens)
d2_tokens = rng.randn(5, 64)
d2_tokens = d2_tokens / np.linalg.norm(d2_tokens, axis=1, keepdims=True)

print(f"MaxSim(query, related_doc): {maxsim_score(q_tokens, d1_tokens):.3f}")
print(f"MaxSim(query, random_doc):  {maxsim_score(q_tokens, d2_tokens):.3f}")
# MaxSim preserves fine-grained token matches that
# single-vector compression would average away
