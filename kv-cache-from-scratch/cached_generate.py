import numpy as np
from attention_setup import W_q, W_k, W_v, d_head, d_model, attention

def cached_generate(embeddings, num_new_tokens):
    """Generate tokens with KV caching — only project the new token."""
    # Prefill: process the entire prompt at once
    prompt = np.array(embeddings)
    K_cache = prompt @ W_k    # (prompt_len, d_head)
    V_cache = prompt @ W_v    # (prompt_len, d_head)
    Q_all = prompt @ W_q
    flops = 3 * len(embeddings)  # initial projections

    out = attention(Q_all, K_cache, V_cache)
    next_emb = out[-1]

    seq = list(embeddings) + [next_emb]

    for step in range(num_new_tokens - 1):
        x_new = next_emb.reshape(1, -1)   # (1, d_model)

        # Project ONLY the new token — one vector, not the whole sequence
        q_new = x_new @ W_q    # (1, d_head)
        k_new = x_new @ W_k    # (1, d_head)
        v_new = x_new @ W_v    # (1, d_head)
        flops += 3             # 3 projections × 1 token

        # Append to cache
        K_cache = np.vstack([K_cache, k_new])  # grows by 1 row
        V_cache = np.vstack([V_cache, v_new])

        # Attention: new query against ALL cached keys and values
        scores = q_new @ K_cache.T / np.sqrt(d_head)  # (1, cache_len)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        next_emb = (weights @ V_cache).flatten()

        seq.append(next_emb)

    return np.array(seq), flops

if __name__ == "__main__":
    prompt = [np.random.randn(d_model) for _ in range(10)]
    seq, flops = cached_generate(prompt, 100)
    print(f"Cached: generated {len(seq)} tokens, {flops} projection ops")
