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

# --- Beam search ---
def beam_search(start_token, num_tokens=10, beam_width=3):
    """Find the most probable sequence by exploring multiple paths."""
    # Each beam: (token_list, cumulative_log_probability)
    beams = [([start_token], 0.0)]

    for step in range(num_tokens):
        candidates = []
        for tokens, score in beams:
            logits = get_logits(tokens[-1])
            probs = softmax(logits)
            log_probs = np.log(probs + 1e-10)

            for i, lp in enumerate(log_probs):
                new_seq = tokens + [vocab[i]]
                candidates.append((new_seq, score + lp))

        # Keep only the top beam_width candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    return beams[0][0]  # return the highest-scoring beam

print(" ".join(beam_search("the", num_tokens=12, beam_width=3)))
# the cat sat on the cat sat on the cat sat on the
