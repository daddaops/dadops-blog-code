"""Instrumental Variables (2SLS) estimation from scratch.

Estimates the causal return to education on earnings using proximity to college
as an instrument. Ability is an unobserved confounder that biases OLS upward.
True causal return is $3K/year; OLS overestimates due to ability bias.
"""
import numpy as np

np.random.seed(42)
n = 2000

# Unobserved confounder: innate ability
ability = np.random.normal(0, 1, n)

# Instrument: proximity to college (0 = far, 1 = near)
# Independent of ability (geographic accident)
near_college = (np.random.random(n) < 0.4).astype(float)

# Education (affected by ability AND proximity)
education = 12 + 2 * ability + 1.5 * near_college + np.random.normal(0, 1, n)

# Earnings (affected by education AND ability directly)
true_return = 3.0  # true causal return to education: $3K per year
earnings = 20 + true_return * education + 5 * ability + np.random.normal(0, 5, n)

# OLS regression (biased: ability confounds education-earnings)
X_ols = np.column_stack([np.ones(n), education])
beta_ols = np.linalg.lstsq(X_ols, earnings, rcond=None)[0]

# 2SLS: Instrumental Variables using college proximity
# Stage 1: Regress education on instrument
X_iv = np.column_stack([np.ones(n), near_college])
gamma = np.linalg.lstsq(X_iv, education, rcond=None)[0]
education_hat = X_iv @ gamma  # predicted education from instrument

# Stage 2: Regress earnings on predicted education
X_2sls = np.column_stack([np.ones(n), education_hat])
beta_2sls = np.linalg.lstsq(X_2sls, earnings, rcond=None)[0]

# First-stage F-statistic (instrument relevance check)
ss_resid = np.sum((education - X_iv @ gamma) ** 2)
ss_total = np.sum((education - education.mean()) ** 2)
r_squared = 1 - ss_resid / ss_total
f_stat = (r_squared / 1) / ((1 - r_squared) / (n - 2))

print(f"True causal return to education: ${true_return:.2f}K/year")
print(f"OLS estimate:                    ${beta_ols[1]:.2f}K/year  (biased upward)")
print(f"2SLS IV estimate:                ${beta_2sls[1]:.2f}K/year  (consistent)")
print(f"\nOLS bias: ability is a confounder (smarter people get more education")
print(f"AND earn more, inflating the apparent return to education)")
print(f"\nFirst-stage F-statistic: {f_stat:.1f}  ({'Strong' if f_stat > 10 else 'Weak'} instrument)")
print(f"Rule of thumb: F > 10 indicates a strong instrument")
