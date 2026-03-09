import numpy as np
from attention_setup import W_q, W_k, W_v, d_head, d_model

def apply_rope(x, position):
    """Minimal RoPE: rotate pairs of dimensions by position-dependent angles."""
    d = len(x)
    rotated = x.copy()
    for i in range(0, d, 2):
        freq = 1.0 / (10000.0 ** (i / d))
        angle = position * freq
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated[i] = x[i] * cos_a - x[i + 1] * sin_a
        rotated[i + 1] = x[i] * sin_a + x[i + 1] * cos_a
    return rotated

def decode_step_with_rope(new_emb, position, K_cache, V_cache):
    """RoPE is applied to K before it enters the cache."""
    q = new_emb @ W_q
    k = new_emb @ W_k
    v = new_emb @ W_v

    # Apply RoPE rotation based on this token's absolute position
    q = apply_rope(q, position)  # rotate query by position angle
    k = apply_rope(k, position)  # rotate key by position angle

    # Cache the ROTATED key — position is baked in permanently
    K_cache = np.vstack([K_cache, k.reshape(1, -1)])
    V_cache = np.vstack([V_cache, v.reshape(1, -1)])  # V is NOT rotated

    # Attention: rotated query against rotated keys
    # The dot product Q_s · K_t naturally captures relative distance (s - t)
    scores = q.reshape(1, -1) @ K_cache.T / np.sqrt(d_head)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)

    return (weights @ V_cache).flatten(), K_cache, V_cache

if __name__ == "__main__":
    # Build initial cache from a 5-token prompt with RoPE
    prompt = [np.random.randn(d_model) for _ in range(5)]
    K_cache = np.zeros((0, d_head))
    V_cache = np.zeros((0, d_head))

    last_out = None
    for pos, emb in enumerate(prompt):
        k = apply_rope(emb @ W_k, pos)
        v = emb @ W_v
        K_cache = np.vstack([K_cache, k.reshape(1, -1)]) if K_cache.size else k.reshape(1, -1)
        V_cache = np.vstack([V_cache, v.reshape(1, -1)]) if V_cache.size else v.reshape(1, -1)
        last_out = emb

    # Decode 10 more tokens
    for i in range(10):
        last_out, K_cache, V_cache = decode_step_with_rope(
            last_out, len(prompt) + i, K_cache, V_cache)

    print(f"RoPE-cached generation: cache shape = {K_cache.shape}")
