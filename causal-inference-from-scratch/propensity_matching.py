"""Propensity Score Matching (PSM) from scratch.

Implements logistic regression via gradient ascent, nearest-neighbor matching
on propensity scores, and estimates the Average Treatment Effect on the Treated (ATT).
True ATE is $4K; naive estimate is biased by self-selection.
"""
import numpy as np

np.random.seed(42)
n = 1000

# Confounders
experience = np.random.exponential(5, n)
education = np.random.normal(14, 2, n)  # years of schooling

# Treatment assignment (self-selection based on confounders)
logit = -1.5 + 0.1 * experience + 0.15 * education
prob_treat = 1 / (1 + np.exp(-logit))
treatment = (np.random.random(n) < prob_treat).astype(int)

# Potential outcomes
y0 = 25 + 1.8 * experience + 2.5 * education + np.random.normal(0, 4, n)
y1 = y0 + 4  # true ATE = $4K
y_obs = treatment * y1 + (1 - treatment) * y0

# Step 1: Estimate propensity scores via logistic regression
X = np.column_stack([np.ones(n), experience, education])
beta = np.zeros(3)
lr = 0.01
for _ in range(1000):  # gradient ascent
    z = X @ beta
    p = 1 / (1 + np.exp(-np.clip(z, -20, 20)))
    gradient = X.T @ (treatment - p) / n
    beta += lr * gradient

propensity = 1 / (1 + np.exp(-np.clip(X @ beta, -20, 20)))

# Step 2: Nearest-neighbor matching on propensity score
treated_idx = np.where(treatment == 1)[0]
control_idx = np.where(treatment == 0)[0]

matched_treated = []
matched_control = []
for t in treated_idx:
    distances = np.abs(propensity[t] - propensity[control_idx])
    best_match = control_idx[np.argmin(distances)]
    matched_treated.append(y_obs[t])
    matched_control.append(y_obs[best_match])

# Step 3: Estimate ATT from matched pairs
naive_ate = y_obs[treatment == 1].mean() - y_obs[treatment == 0].mean()
matched_att = np.mean(np.array(matched_treated) - np.array(matched_control))
true_ate = 4.0

print(f"True ATE:         ${true_ate:.2f}K")
print(f"Naive estimate:   ${naive_ate:.2f}K  (biased by self-selection)")
print(f"PSM estimate:     ${matched_att:.2f}K  (bias reduced!)")
print(f"\nBias reduction:   {abs(naive_ate - true_ate) - abs(matched_att - true_ate):.2f}K closer to truth")
print(f"Treated group:    {len(treated_idx)} workers")
print(f"Matched pairs:    {len(matched_treated)}")
