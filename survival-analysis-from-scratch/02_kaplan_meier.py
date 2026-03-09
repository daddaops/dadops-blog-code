"""Kaplan-Meier estimator with Greenwood standard errors."""
import numpy as np


def kaplan_meier(times, events):
    """
    Kaplan-Meier estimator with Greenwood standard errors.

    Args:
        times: array of observed times (event or censoring)
        events: array of 0/1 (1 = event occurred, 0 = censored)

    Returns:
        km_times: event time points (with 0 prepended)
        km_surv: survival probabilities at each time
        km_se: standard errors (Greenwood's formula)
    """
    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]

    # Find unique event times (only times where events occur)
    event_mask = e_sorted == 1
    event_times = np.unique(t_sorted[event_mask])

    n_total = len(times)
    surv = 1.0
    greenwood_sum = 0.0

    km_times = [0.0]
    km_surv = [1.0]
    km_se = [0.0]

    # Track position in sorted data for risk set counting
    n_at_risk = n_total
    idx = 0

    for t_i in event_times:
        # Remove subjects censored or with events before this time
        while idx < n_total and t_sorted[idx] < t_i:
            n_at_risk -= 1
            idx += 1

        # Count events at this exact time
        d_i = 0
        n_censored_at_t = 0
        while idx < n_total and t_sorted[idx] == t_i:
            if e_sorted[idx] == 1:
                d_i += 1
            else:
                n_censored_at_t += 1
            idx += 1

        # Kaplan-Meier update
        surv *= (1.0 - d_i / n_at_risk)

        # Greenwood variance accumulator
        if n_at_risk > d_i:
            greenwood_sum += d_i / (n_at_risk * (n_at_risk - d_i))

        km_times.append(t_i)
        km_surv.append(surv)
        km_se.append(surv * np.sqrt(greenwood_sum))

        # Remove events and censored-at-same-time from risk set
        n_at_risk -= (d_i + n_censored_at_t)

    return np.array(km_times), np.array(km_surv), np.array(km_se)


if __name__ == "__main__":
    # Example: 10 patients
    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    events = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 0 = censored

    km_t, km_s, km_se = kaplan_meier(times, events)
    for t, s, se in zip(km_t, km_s, km_se):
        if t > 0:
            print(f"t={t:.0f}: S(t)={s:.3f} +/- {se:.3f}")
