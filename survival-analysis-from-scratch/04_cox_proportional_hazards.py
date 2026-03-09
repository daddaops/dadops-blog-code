"""Cox Proportional Hazards model with Breslow baseline hazard estimator."""
import numpy as np


def cox_gradient(beta, X, times, events):
    """Gradient of negative log partial likelihood."""
    order = np.argsort(-times)
    X_s = X[order]
    e_s = events[order]

    eta = X_s @ beta
    theta = np.exp(eta - np.max(eta))

    cum_theta = np.cumsum(theta)
    cum_theta_x = np.cumsum(theta[:, None] * X_s, axis=0)

    # Weighted mean covariate in each risk set
    x_bar = cum_theta_x / cum_theta[:, None]

    grad = np.sum(e_s[:, None] * (X_s - x_bar), axis=0)
    return -grad


def fit_cox(X, times, events, lr=0.01, max_iter=1000, tol=1e-8):
    """Fit Cox PH model via gradient descent."""
    beta = np.zeros(X.shape[1])
    for i in range(max_iter):
        g = cox_gradient(beta, X, times, events)
        beta -= lr * g
        if np.max(np.abs(g)) < tol:
            break
    return beta


def breslow_baseline_hazard(beta, X, times, events):
    """Breslow estimator for cumulative baseline hazard H0(t)."""
    event_times_unique = np.sort(np.unique(times[events == 1]))
    risk_scores = np.exp(X @ beta)

    H0_times, H0_values = [], []
    H0 = 0.0
    for t_i in event_times_unique:
        d_i = np.sum((times == t_i) & (events == 1))
        risk_sum = np.sum(risk_scores[times >= t_i])
        H0 += d_i / risk_sum
        H0_times.append(t_i)
        H0_values.append(H0)

    return np.array(H0_times), np.array(H0_values)


if __name__ == "__main__":
    # Example: age and treatment as covariates
    np.random.seed(99)
    n = 200
    age = np.random.normal(60, 10, n)
    treatment = np.random.binomial(1, 0.5, n)
    X = np.column_stack([(age - 60) / 10, treatment])  # standardized

    # Generate survival times: higher age = higher risk, treatment = lower risk
    true_beta = np.array([0.5, -0.8])
    baseline_scale = 10
    T = np.random.exponential(baseline_scale * np.exp(-X @ true_beta))
    C = np.random.uniform(5, 25, n)  # random censoring
    observed_time = np.minimum(T, C)
    event = (T <= C).astype(int)

    beta_hat = fit_cox(X, observed_time, event)
    print(f"True beta:      [{true_beta[0]:.2f}, {true_beta[1]:.2f}]")
    print(f"Estimated beta: [{beta_hat[0]:.2f}, {beta_hat[1]:.2f}]")
    print(f"Hazard ratios:  age_10yr={np.exp(beta_hat[0]):.2f}, "
          f"treatment={np.exp(beta_hat[1]):.2f}")

    # Breslow estimator for baseline survival
    H0_t, H0_v = breslow_baseline_hazard(beta_hat, X, observed_time, event)
    S0_5 = np.exp(-H0_v[np.searchsorted(H0_t, 5, side='right') - 1])
    S0_10 = np.exp(-H0_v[np.searchsorted(H0_t, 10, side='right') - 1])
    print(f"Baseline survival: S0(5)={S0_5:.3f}, S0(10)={S0_10:.3f}")
