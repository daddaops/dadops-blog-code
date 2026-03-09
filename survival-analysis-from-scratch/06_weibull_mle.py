"""Weibull MLE for censored survival data."""
import numpy as np
from scipy.optimize import minimize


def weibull_neg_log_likelihood(params, times, events):
    """
    Negative log-likelihood for Weibull model with censoring.
    params: [log_k, log_lambda] (log-transformed for positivity)
    """
    k = np.exp(params[0])
    lam = np.exp(params[1])

    n_events = np.sum(events)
    log_lik = (n_events * np.log(k)
               - n_events * k * np.log(lam)
               + (k - 1) * np.sum(events * np.log(times))
               - np.sum((times / lam) ** k))
    return -log_lik


def fit_weibull(times, events):
    """MLE for Weibull distribution with censored data."""
    # Initial guess: k=1 (exponential), lambda=median time
    x0 = [np.log(1.0), np.log(np.median(times))]

    result = minimize(weibull_neg_log_likelihood, x0,
                      args=(times, events), method='Nelder-Mead')

    k_hat = np.exp(result.x[0])
    lam_hat = np.exp(result.x[1])
    return k_hat, lam_hat


if __name__ == "__main__":
    # Generate data (same as Cox model script)
    np.random.seed(99)
    n = 200
    age = np.random.normal(60, 10, n)
    treatment = np.random.binomial(1, 0.5, n)
    X = np.column_stack([(age - 60) / 10, treatment])

    true_beta = np.array([0.5, -0.8])
    baseline_scale = 10
    T = np.random.exponential(baseline_scale * np.exp(-X @ true_beta))
    C = np.random.uniform(5, 25, n)
    observed_time = np.minimum(T, C)
    event = (T <= C).astype(int)

    # Fit Weibull to our simulated data (all subjects combined)
    k_hat, lam_hat = fit_weibull(observed_time, event)
    print(f"Weibull MLE: shape k={k_hat:.2f}, scale lambda={lam_hat:.2f}")
    print(f"Median survival: {lam_hat * np.log(2)**(1/k_hat):.1f} time units")
