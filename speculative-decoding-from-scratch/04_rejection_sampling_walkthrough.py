"""Step-by-step walkthrough of rejection sampling math proving losslessness."""

import numpy as np

# Vocabulary: ["the", "cat", "sat", "dog"]
p = np.array([0.50, 0.20, 0.10, 0.20])  # target model
q = np.array([0.40, 0.30, 0.20, 0.10])  # draft model

# Draft model sampled "cat" (index 1) with q("cat") = 0.30
draft_token = 1

# Accept probability: min(1, p("cat")/q("cat")) = min(1, 0.20/0.30)
accept_prob = min(1.0, p[draft_token] / q[draft_token])
print(f"Accept probability for 'cat': {accept_prob:.3f}")   # 0.667

# If rejected, sample from residual:
residual = np.maximum(0, p - q)
print(f"Raw residual: {residual}")
# [max(0, 0.50-0.40), max(0, 0.20-0.30), max(0, 0.10-0.20), max(0, 0.20-0.10)]
# = [0.10, 0.00, 0.00, 0.10]

residual_norm = residual / residual.sum()
print(f"Correction distribution: {residual_norm}")
# = [0.50, 0.00, 0.00, 0.50]
# Correction chooses "the" or "dog" -- tokens the draft underweighted

# Verify losslessness for each token:
for i, word in enumerate(["the", "cat", "sat", "dog"]):
    total = min(p[i], q[i]) + max(0, p[i] - q[i])
    print(f"P('{word}') = min({p[i]:.2f}, {q[i]:.2f}) + "
          f"max(0, {p[i]:.2f}-{q[i]:.2f}) = {total:.2f} == p = {p[i]:.2f} ✓")
