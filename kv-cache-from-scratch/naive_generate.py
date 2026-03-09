import numpy as np
from attention_setup import W_q, W_k, W_v, d_head, d_model, attention

def naive_generate(embeddings, num_new_tokens):
    """Generate tokens by recomputing full attention each step."""
    seq = list(embeddings)  # start with prompt embeddings
    flops = 0

    for step in range(num_new_tokens):
        x = np.array(seq)        # shape: (current_len, d_model)
        current_len = len(seq)

        # Project ALL tokens — even the ones we projected last step
        Q = x @ W_q              # (current_len, d_head)
        K = x @ W_k              # (current_len, d_head)
        V = x @ W_v              # (current_len, d_head)
        flops += 3 * current_len  # 3 projections × current_len tokens

        out = attention(Q, K, V)  # (current_len, d_head)

        # Use the last position's output as the "next token" embedding
        next_token = out[-1]
        seq.append(next_token)

    return np.array(seq), flops

if __name__ == "__main__":
    prompt = [np.random.randn(d_model) for _ in range(10)]
    seq, flops = naive_generate(prompt, 100)
    print(f"Naive: generated {len(seq)} tokens, {flops} projection ops")
