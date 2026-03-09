"""Harrell's C-index for survival model discrimination."""
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


def concordance_index(risk_scores, times, events):
    """
    Harrell's C-index for survival model discrimination.

    Args:
        risk_scores: predicted risk (higher = more risk)
        times: observed times
        events: event indicators (1=event, 0=censored)

    Returns:
        C-index between 0 and 1
    """
    n = len(times)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Check if pair is comparable
            if times[i] < times[j]:
                if events[i] == 0:
                    continue  # earlier subject censored, not comparable
                # Subject i had event first -- should have higher risk
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                else:
                    tied += 1
            elif times[j] < times[i]:
                if events[j] == 0:
                    continue
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] < risk_scores[i]:
                    discordant += 1
                else:
                    tied += 1
            else:  # tied times
                if events[i] == 1 and events[j] == 1:
                    tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


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

    # Fit Cox model and evaluate
    beta_hat = fit_cox(X, observed_time, event)
    risk = np.exp(X @ beta_hat)
    c_idx = concordance_index(risk, observed_time, event)
    print(f"C-index: {c_idx:.3f}")
