"""Metropolis-Hastings MCMC from Scratch — Gaussian parameter inference.

Implements a general-purpose Metropolis-Hastings sampler and uses it to
infer the mean and standard deviation of a Gaussian from data.
"""
import numpy as np

def metropolis_hastings(log_posterior_fn, initial, n_samples, step_size):
    """General-purpose Metropolis-Hastings sampler."""
    current = np.array(initial, dtype=float)
    samples = np.zeros((n_samples, len(current)))
    log_p_current = log_posterior_fn(current)
    n_accepted = 0

    for i in range(n_samples):
        # Propose new position
        proposal = current + np.random.normal(0, step_size, size=len(current))
        log_p_proposal = log_posterior_fn(proposal)

        # Accept/reject
        log_ratio = log_p_proposal - log_p_current
        if np.log(np.random.random()) < log_ratio:
            current = proposal
            log_p_current = log_p_proposal
            n_accepted += 1

        samples[i] = current

    print(f"Acceptance rate: {n_accepted / n_samples:.2%}")
    return samples

# Problem: infer the mean and std of a Gaussian from data
np.random.seed(42)
true_mu, true_sigma = 3.0, 1.5
data = np.random.normal(true_mu, true_sigma, 50)

def log_posterior(params):
    """Log-posterior for Gaussian mean and log-std with weak priors."""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    # Likelihood: product of Gaussian PDFs (in log space: sum)
    log_lik = -0.5 * np.sum(((data - mu) / sigma)**2) - len(data) * np.log(sigma)
    # Priors: N(0, 10) on mu, N(0, 2) on log_sigma
    log_prior = -0.5 * (mu / 10)**2 - 0.5 * (log_sigma / 2)**2
    return log_lik + log_prior

samples = metropolis_hastings(log_posterior, [0.0, 0.0], n_samples=10000, step_size=0.15)
# Acceptance rate: ~45%

# Discard burn-in (first 2000 samples)
posterior_samples = samples[2000:]
mu_samples = posterior_samples[:, 0]
sigma_samples = np.exp(posterior_samples[:, 1])

print(f"Posterior mean of mu:    {mu_samples.mean():.2f} +/- {mu_samples.std():.2f}")
print(f"Posterior mean of sigma: {sigma_samples.mean():.2f} +/- {sigma_samples.std():.2f}")
# mu ~ 3.0, sigma ~ 1.5 — recovers the true parameters!
