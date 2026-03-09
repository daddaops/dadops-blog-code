import numpy as np

def softmax(logits, T=1.0):
    """Softmax with temperature scaling."""
    scaled = logits / T
    shifted = scaled - np.max(scaled)  # numerical stability
    exps = np.exp(shifted)
    return exps / np.sum(exps)

# Teacher's raw logits for an image of handwritten "2"
# These are the values BEFORE softmax — the teacher's "raw opinion"
teacher_logits = np.array([
    -1.5,   # digit 0 — round, no sharp angles like a 2
     0.2,   # digit 1 — has a vertical stroke, some similarity
     5.0,   # digit 2 — correct class, high logit
     2.1,   # digit 3 — similar curves! dark knowledge here
    -1.0,   # digit 4 — angular, not similar
     0.8,   # digit 5 — top half has some resemblance
    -0.5,   # digit 6 — round bottom, some similarity
     1.6,   # digit 7 — shares top-left stroke! more dark knowledge
    -0.7,   # digit 8 — roundish, not very similar
     0.1,   # digit 9 — top loop vaguely like a 2
])

if __name__ == "__main__":
    # Hard label: "this is a 2"
    hard_label = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    # Teacher's soft predictions at standard temperature (T=1)
    soft_preds = softmax(teacher_logits, T=1.0)

    print("Hard label:     ", hard_label)
    print("Teacher (T=1):  ", [f"{p:.4f}" for p in soft_preds])
    # Hard label:      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # Teacher (T=1):   ['0.0013', '0.0073', '0.8846', '0.0487', '0.0022',
    #                    '0.0133', '0.0036', '0.0295', '0.0030', '0.0066']
