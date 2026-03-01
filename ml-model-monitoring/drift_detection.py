"""
Code Block 1: Data drift detection metrics.

From: https://dadops.dev/blog/ml-model-monitoring/

compute_psi() — Population Stability Index between two distributions.
ks_statistic() — Kolmogorov-Smirnov statistic (max CDF distance).
js_divergence() — Jensen-Shannon divergence (symmetric, bounded [0,1]).

No external dependencies required.
"""

import math


def compute_psi(reference, production, bins=10):
    """Population Stability Index between two distributions."""
    min_val = min(min(reference), min(production))
    max_val = max(max(reference), max(production))

    def bin_counts(data):
        counts = [0] * bins
        for v in data:
            idx = min(int((v - min_val) / (max_val - min_val) * bins), bins - 1)
            counts[idx] += 1
        return [c / len(data) + 1e-8 for c in counts]  # avoid log(0)

    ref_pct = bin_counts(reference)
    prod_pct = bin_counts(production)
    psi = sum((p - r) * math.log(p / r) for r, p in zip(ref_pct, prod_pct))
    return psi


def ks_statistic(reference, production):
    """Kolmogorov-Smirnov statistic (max CDF distance)."""
    all_vals = sorted(set(reference + production))
    n_ref, n_prod = len(reference), len(production)
    ref_sorted = sorted(reference)
    prod_sorted = sorted(production)

    max_dist = 0.0
    ref_idx, prod_idx = 0, 0
    for val in all_vals:
        while ref_idx < n_ref and ref_sorted[ref_idx] <= val:
            ref_idx += 1
        while prod_idx < n_prod and prod_sorted[prod_idx] <= val:
            prod_idx += 1
        cdf_ref = ref_idx / n_ref
        cdf_prod = prod_idx / n_prod
        max_dist = max(max_dist, abs(cdf_ref - cdf_prod))
    return max_dist


def js_divergence(reference, production, bins=10):
    """Jensen-Shannon divergence (symmetric, bounded [0, 1])."""
    min_val = min(min(reference), min(production))
    max_val = max(max(reference), max(production))

    def to_hist(data):
        counts = [0] * bins
        for v in data:
            idx = min(int((v - min_val) / (max_val - min_val) * bins), bins - 1)
            counts[idx] += 1
        return [c / len(data) + 1e-8 for c in counts]

    p, q = to_hist(reference), to_hist(production)
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    kl_pm = sum(pi * math.log(pi / mi) for pi, mi in zip(p, m))
    kl_qm = sum(qi * math.log(qi / mi) for qi, mi in zip(q, m))
    return (kl_pm + kl_qm) / 2


if __name__ == "__main__":
    import random

    print("=== Drift Detection Demo ===\n")

    # Simulate: training data (normal) vs production data (shifted)
    random.seed(42)
    reference = [random.gauss(50, 10) for _ in range(1000)]
    production_stable = [random.gauss(50, 10) for _ in range(1000)]
    production_drifted = [random.gauss(58, 12) for _ in range(1000)]

    print("Stable production (no drift):")
    print(f"  PSI = {compute_psi(reference, production_stable):.4f}")
    print(f"  KS  = {ks_statistic(reference, production_stable):.4f}")
    print(f"  JS  = {js_divergence(reference, production_stable):.4f}")

    print("Drifted production (mean +8, std +2):")
    print(f"  PSI = {compute_psi(reference, production_drifted):.4f}")
    print(f"  KS  = {ks_statistic(reference, production_drifted):.4f}")
    print(f"  JS  = {js_divergence(reference, production_drifted):.4f}")
