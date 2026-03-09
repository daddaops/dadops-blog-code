import numpy as np
from attention_setup import W_q, W_k, W_v, d_head, d_model, attention

def prefill(prompt_embeddings):
    """Process the full prompt in parallel, return the KV cache."""
    x = np.array(prompt_embeddings)
    K_cache = x @ W_k       # (prompt_len, d_head)
    V_cache = x @ W_v       # (prompt_len, d_head)
    Q = x @ W_q

    out = attention(Q, K_cache, V_cache)
    return K_cache, V_cache, out[-1]

def decode_step(new_embedding, K_cache, V_cache):
    """Generate one token using the KV cache."""
    x = new_embedding.reshape(1, -1)

    q = x @ W_q
    k = x @ W_k
    v = x @ W_v

    K_cache = np.vstack([K_cache, k])
    V_cache = np.vstack([V_cache, v])

    scores = q @ K_cache.T / np.sqrt(d_head)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    output = (weights @ V_cache).flatten()

    return output, K_cache, V_cache

# Usage: clean separation of phases
prompt = [np.random.randn(d_model) for _ in range(10)]
K, V, last_out = prefill(prompt)

for _ in range(50):
    last_out, K, V = decode_step(last_out, K, V)

print(f"Prefill+decode: cache shape = {K.shape}, final output shape = {last_out.shape}")
