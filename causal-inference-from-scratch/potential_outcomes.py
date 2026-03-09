"""Rubin Causal Model / Potential Outcomes framework demo.

Simulates a job training program where less experienced workers (who earn less)
are more likely to enroll -- a classic case of negative selection bias.
The true ATE is $5K, but naive comparison gives the wrong sign.

Expected output:
  True ATE:        $5.00K
  Naive estimate:  $-2.14K   (WRONG SIGN!)
  Selection bias:  $-7.14K
"""
import numpy as np

np.random.seed(42)
n = 500

# X = years of experience (confounder)
experience = np.random.exponential(scale=5, size=n)

# Potential outcomes (God mode -- we see both)
y0 = 30 + 2 * experience + np.random.normal(0, 3, n)   # salary without training
y1 = y0 + 5                                              # salary with training (true ATE = $5K)

# Selection: less experienced workers MORE likely to seek training
prob_treat = 1 / (1 + np.exp(0.5 * (experience - 4)))
treatment = (np.random.random(n) < prob_treat).astype(int)

# Observed outcomes (fundamental problem: we only see one)
y_observed = treatment * y1 + (1 - treatment) * y0

# Naive estimate: compare treated vs untreated means
naive_ate = y_observed[treatment == 1].mean() - y_observed[treatment == 0].mean()

# True ATE (only possible in simulation)
true_ate = (y1 - y0).mean()

# Selection bias decomposition
# E[Y(0)|T=1] - E[Y(0)|T=0]: baseline difference between groups
selection_bias = y0[treatment == 1].mean() - y0[treatment == 0].mean()

print(f"True ATE:        ${true_ate:.2f}K")
print(f"Naive estimate:  ${naive_ate:.2f}K")
print(f"Selection bias:  ${selection_bias:.2f}K")
print(f"Check: naive ≈ ATE + bias: {true_ate:.2f} + ({selection_bias:.2f}) = {true_ate + selection_bias:.2f}")
print(f"\nTreated group avg experience:   {experience[treatment == 1].mean():.1f} years")
print(f"Untreated group avg experience: {experience[treatment == 0].mean():.1f} years")
print("Less experienced workers self-selected into training,")
print("dragging down the treated group's observed outcomes.")
