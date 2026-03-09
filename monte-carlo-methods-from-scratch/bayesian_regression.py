"""MCMC for Bayesian linear regression.

Uses 2D Metropolis-Hastings to sample the posterior over slope
and intercept, recovering credible intervals.
"""
import numpy as np

np.random.seed(42)

# Generate synthetic data: y = 2.5x - 1.0 + noise
n_data = 50
x_data = np.random.uniform(-2, 2, n_data)
y_data = 2.5 * x_data - 1.0 + np.random.normal(0, 0.8, n_data)

sigma = 0.8  # known noise level

def log_posterior(a, b):
    # Priors: a ~ N(0, 10), b ~ N(0, 10)
    log_prior = -0.5 * (a**2 / 100 + b**2 / 100)
    # Likelihood: y_i ~ N(a*x_i + b, sigma^2)
    residuals = y_data - (a * x_data + b)
    log_lik = -0.5 * np.sum(residuals**2) / sigma**2
    return log_prior + log_lik

# 2D Metropolis-Hastings
n_samples = 20_000
step_size = 0.1
chain = np.zeros((n_samples, 2))
chain[0] = [0.0, 0.0]
accepted = 0

for i in range(1, n_samples):
    proposal = chain[i-1] + np.random.normal(0, step_size, 2)
    log_alpha = log_posterior(*proposal) - log_posterior(*chain[i-1])

    if np.log(np.random.uniform()) < log_alpha:
        chain[i] = proposal
        accepted += 1
    else:
        chain[i] = chain[i-1]

burn_in = 2000
posterior = chain[burn_in:]
a_samples, b_samples = posterior[:, 0], posterior[:, 1]

print(f"Acceptance rate: {accepted / n_samples:.1%}")
print(f"Slope a:     {np.mean(a_samples):.3f} +/- {np.std(a_samples):.3f}  (true: 2.500)")
print(f"Intercept b: {np.mean(b_samples):.3f} +/- {np.std(b_samples):.3f}  (true: -1.000)")

# 95% credible intervals
a_ci = np.percentile(a_samples, [2.5, 97.5])
b_ci = np.percentile(b_samples, [2.5, 97.5])
print(f"95% CI for a: [{a_ci[0]:.3f}, {a_ci[1]:.3f}]")
print(f"95% CI for b: [{b_ci[0]:.3f}, {b_ci[1]:.3f}]")
