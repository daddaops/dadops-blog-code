"""Attention head taxonomy: classify synthetic attention patterns by type.

Generates four canonical attention head types (previous-token, induction,
duplicate-token, positional) and computes diagnostic scores to classify them.
"""
import numpy as np


def classify_attention_heads():
    """Generate and classify synthetic attention patterns by head type."""
    seq_len = 8
    tokens = ["The", "cat", "sat", "on", "the", "cat", "saw", "the"]

    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    # Generate 4 attention pattern types
    heads = {}

    # 1. Previous-token head: strong diagonal offset by -1
    score = np.full((seq_len, seq_len), -10.0)
    for i in range(1, seq_len):
        score[i, i - 1] = 5.0
    score[0, 0] = 5.0  # first token attends to itself
    heads["Previous-token"] = softmax(score)

    # 2. Induction head: attends to token AFTER previous occurrence of current token
    score = np.full((seq_len, seq_len), -10.0)
    for i in range(seq_len):
        for j in range(i):
            if tokens[j] == tokens[i] and j + 1 < seq_len:
                score[i, j + 1] = 5.0  # attend to what followed the match
    # Fill remaining with uniform
    for i in range(seq_len):
        if score[i].max() < -5:
            score[i, :i + 1] = 0.0
    heads["Induction"] = softmax(score)

    # 3. Duplicate-token head: attends to previous occurrences of same token
    score = np.full((seq_len, seq_len), -10.0)
    for i in range(seq_len):
        for j in range(i):
            if tokens[j] == tokens[i]:
                score[i, j] = 5.0
        if score[i].max() < -5:
            score[i, :i + 1] = 0.0
    heads["Duplicate-token"] = softmax(score)

    # 4. Positional head: always attends to position 0
    score = np.full((seq_len, seq_len), -10.0)
    score[:, 0] = 5.0
    heads["Positional (BOS)"] = softmax(score)

    # Classify each head using diagnostic scores
    print(f"Tokens: {tokens}\n")
    for name, attn in heads.items():
        # Previous-token score: mean attention on position i-1
        prev_score = np.mean([attn[i, i-1] for i in range(1, seq_len)])
        # Positional score: attention entropy (low = positional)
        entropy = -np.sum(attn * np.log(attn + 1e-10), axis=-1).mean()
        # Duplicate score: attention on matching tokens
        dup_score = np.mean([max(attn[i, j] for j in range(i)
                             if tokens[j] == tokens[i])
                             for i in range(seq_len) if any(
                                 tokens[j] == tokens[i] for j in range(i))])
        print(f"{name:<20} prev={prev_score:.2f}  entropy={entropy:.2f}  dup={dup_score:.2f}")


if __name__ == "__main__":
    classify_attention_heads()
