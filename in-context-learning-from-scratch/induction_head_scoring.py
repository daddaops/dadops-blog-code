import numpy as np

def score_induction_heads(attention_weights, tokens):
    """Score each attention head on the induction pattern."""
    n_heads, seq_len, _ = attention_weights.shape
    scores = np.zeros(n_heads)

    for h in range(n_heads):
        total, count = 0.0, 0
        for pos in range(2, seq_len):
            current_token = tokens[pos]
            for prev_pos in range(1, pos):
                if tokens[prev_pos - 1] == current_token:
                    total += attention_weights[h, pos, prev_pos]
                    count += 1
        scores[h] = total / max(count, 1)

    return scores

# Example: a sequence with a repeated pattern
tokens = [10, 20, 30, 40, 10, 20, 30, 40]  # ABCD ABCD
n_heads, seq_len = 4, len(tokens)

# Simulate: head 0 is random, head 2 is an induction head
np.random.seed(7)
attn = np.random.dirichlet(np.ones(seq_len), size=(n_heads, seq_len))

# Make head 2 attend to the induction target
for pos in range(4, seq_len):
    attn[2, pos, :] = 0.01  # small baseline
    attn[2, pos, pos - 4 + 1] = 0.9  # attend to token after match
    attn[2, pos] /= attn[2, pos].sum()

scores = score_induction_heads(attn, tokens)
for h, s in enumerate(scores):
    marker = " <-- induction head!" if s > 0.3 else ""
    print(f"Head {h}: induction score = {s:.3f}{marker}")
# Head 0: induction score = 0.121
# Head 1: induction score = 0.133
# Head 2: induction score = 0.928 <-- induction head!
# Head 3: induction score = 0.116
