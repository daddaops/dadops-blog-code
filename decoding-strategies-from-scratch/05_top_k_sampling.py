import numpy as np

# --- Shared setup from Block 1 ---
vocab = ["the", "cat", "dog", "sat", "ran", "on", "a", "big",
         "small", "mat", "park", "happy", "and", "was", "red"]

transitions = {
    "the":   {"cat": 3.2, "dog": 2.8, "big": 1.5, "red": 1.2,
              "small": 1.0, "mat": 0.8, "park": 0.7},
    "cat":   {"sat": 3.5, "ran": 2.8, "was": 2.0, "and": 1.0},
    "dog":   {"ran": 3.0, "sat": 2.0, "was": 2.0, "and": 1.5},
    "sat":   {"on": 3.5, "and": 1.0, "happy": 0.3},
    "ran":   {"on": 1.5, "and": 1.0, "a": 0.5},
    "on":    {"the": 3.5, "a": 2.5, "big": 0.5},
    "a":     {"big": 2.5, "small": 2.0, "red": 1.8, "happy": 1.2,
              "cat": 0.8, "dog": 0.8, "mat": 0.5},
    "big":   {"cat": 2.5, "dog": 2.0, "mat": 1.5, "red": 0.8},
    "small": {"cat": 2.5, "dog": 2.0, "mat": 1.5, "red": 0.8},
    "mat":   {"and": 2.5, "was": 2.0},
    "park":  {"and": 2.0, "was": 1.5, "on": 0.8},
    "happy": {"cat": 2.5, "dog": 2.0, "and": 1.5},
    "and":   {"the": 3.0, "a": 2.5, "ran": 1.0, "sat": 0.8},
    "was":   {"happy": 2.0, "big": 1.5, "small": 1.2, "red": 1.0,
              "on": 0.5},
    "red":   {"cat": 2.5, "dog": 2.0, "mat": 1.8},
}

def get_logits(token):
    logits = np.full(len(vocab), -1.0)
    if token in transitions:
        for word, value in transitions[token].items():
            logits[vocab.index(word)] = value
    return logits

def softmax(logits, temperature=1.0):
    scaled = logits / temperature
    exp_vals = np.exp(scaled - np.max(scaled))
    return exp_vals / exp_vals.sum()

# --- Top-k sampling ---
def top_k_sample(start_token, num_tokens=15, k=5, temperature=1.0):
    """Sample from only the top-k most probable tokens."""
    tokens = [start_token]
    for _ in range(num_tokens):
        logits = get_logits(tokens[-1])
        probs = softmax(logits, temperature=temperature)

        # Zero out everything except the top k
        top_k_idx = np.argsort(probs)[-k:]
        filtered = np.zeros_like(probs)
        filtered[top_k_idx] = probs[top_k_idx]
        filtered /= filtered.sum()  # renormalize

        next_idx = np.random.choice(len(vocab), p=filtered)
        tokens.append(vocab[next_idx])
    return tokens

np.random.seed(7)
print("k=3:", " ".join(top_k_sample("the", 15, k=3)))
# k=3: the cat ran and the dog sat on a big cat sat on a big

print("k=8:", " ".join(top_k_sample("the", 15, k=8)))
# k=8: the cat ran and a red dog was big mat and the big dog sat on
