"""Weibull hazard, cumulative hazard, and survival functions."""
import numpy as np


def weibull_hazard(t, k, lam):
    """Instantaneous hazard h(t) for Weibull distribution."""
    return (k / lam) * (t / lam) ** (k - 1)


def weibull_cumulative_hazard(t, k, lam):
    """Cumulative hazard H(t) for Weibull distribution."""
    return (t / lam) ** k


def weibull_survival(t, k, lam):
    """Survival function S(t) = exp(-H(t))."""
    return np.exp(-weibull_cumulative_hazard(t, k, lam))


if __name__ == "__main__":
    # Verify: S(t) = exp(-H(t)) for all parameter settings
    t = np.linspace(0.01, 15, 200)
    for k, lam in [(0.5, 5), (1.0, 5), (2.0, 5), (1.5, 10)]:
        S = weibull_survival(t, k, lam)
        H = weibull_cumulative_hazard(t, k, lam)
        assert np.allclose(S, np.exp(-H)), f"Failed for k={k}, lam={lam}"

    # Show hazard shapes
    # k=0.5: h(t) decreases -- early risk, then safer
    # k=1.0: h(t) = 0.2 constant -- memoryless (exponential)
    # k=2.0: h(t) increases linearly -- aging/wear-out
    for k in [0.5, 1.0, 2.0]:
        h_vals = weibull_hazard(t, k, lam=5)
        print(f"k={k}: h(1)={h_vals[13]:.3f}, h(5)={h_vals[66]:.3f}, h(10)={h_vals[133]:.3f}")
