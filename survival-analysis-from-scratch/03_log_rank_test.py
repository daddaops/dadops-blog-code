"""Two-sample log-rank test comparing survival between groups."""
import numpy as np
from scipy.stats import chi2


def log_rank_test(times, events, groups):
    """
    Two-sample log-rank test comparing survival between groups.

    Args:
        times: observed times for all subjects
        events: event indicators (1=event, 0=censored)
        groups: group labels (0 or 1)

    Returns:
        chi2_stat: test statistic
        p_value: p-value from chi-squared(1) distribution
    """
    event_times = np.unique(times[events == 1])

    O_minus_E = 0.0
    V_total = 0.0

    for t_j in event_times:
        # Count at-risk and events in each group at this time
        at_risk_1 = np.sum((times >= t_j) & (groups == 0))
        at_risk_2 = np.sum((times >= t_j) & (groups == 1))
        n_j = at_risk_1 + at_risk_2
        if n_j == 0:
            continue

        d_1j = np.sum((times == t_j) & (events == 1) & (groups == 0))
        d_2j = np.sum((times == t_j) & (events == 1) & (groups == 1))
        d_j = d_1j + d_2j

        # Expected events in group 1 under H0
        E_1j = at_risk_1 * d_j / n_j
        O_minus_E += d_1j - E_1j

        # Hypergeometric variance
        if n_j > 1:
            V_j = (at_risk_1 * at_risk_2 * d_j * (n_j - d_j)) / (n_j**2 * (n_j - 1))
            V_total += V_j

    chi2_stat = O_minus_E**2 / V_total if V_total > 0 else 0
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


if __name__ == "__main__":
    # Example: treatment vs control
    np.random.seed(42)
    times_ctrl = np.random.exponential(scale=5.0, size=30)
    times_trt = np.random.exponential(scale=8.0, size=30)
    all_times = np.concatenate([times_ctrl, times_trt])
    all_events = np.ones(60, dtype=int)  # no censoring for simplicity
    all_groups = np.array([0]*30 + [1]*30)

    stat, pval = log_rank_test(all_times, all_events, all_groups)
    print(f"Chi-squared = {stat:.2f}, p-value = {pval:.4f}")
