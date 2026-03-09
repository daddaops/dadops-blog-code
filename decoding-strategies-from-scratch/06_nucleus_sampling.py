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

# --- Nucleus (top-p) sampling ---
def nucleus_sample(start_token, num_tokens=15, p=0.9, temperature=1.0):
    """Sample from the smallest token set whose cumulative probability exceeds p."""
    tokens = [start_token]
    for _ in range(num_tokens):
        logits = get_logits(tokens[-1])
        probs = softmax(logits, temperature=temperature)

        # Sort by probability, descending
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]

        # Find nucleus: smallest set with cumulative prob >= p
        cumulative = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative, p) + 1

        # Keep only nucleus tokens
        nucleus_idx = sorted_idx[:cutoff]
        filtered = np.zeros_like(probs)
        filtered[nucleus_idx] = probs[nucleus_idx]
        filtered /= filtered.sum()

        next_idx = np.random.choice(len(vocab), p=filtered)
        tokens.append(vocab[next_idx])
    return tokens

np.random.seed(7)
print("p=0.9:", " ".join(nucleus_sample("the", 15, p=0.9)))
# p=0.9: the cat ran a mat was happy dog sat on the dog and the cat sat
