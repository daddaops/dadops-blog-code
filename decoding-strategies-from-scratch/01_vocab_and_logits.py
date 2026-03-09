import numpy as np

vocab = ["the", "cat", "dog", "sat", "ran", "on", "a", "big",
         "small", "mat", "park", "happy", "and", "was", "red"]

# Transition logits: given a word, how likely is each next word?
# Only strong connections listed — everything else gets -1.0
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
    """Return logits for the next token. Unknown transitions get -1.0."""
    logits = np.full(len(vocab), -1.0)
    if token in transitions:
        for word, value in transitions[token].items():
            logits[vocab.index(word)] = value
    return logits

def softmax(logits, temperature=1.0):
    """Convert logits to probabilities with temperature scaling."""
    scaled = logits / temperature
    exp_vals = np.exp(scaled - np.max(scaled))  # numerical stability
    return exp_vals / exp_vals.sum()
