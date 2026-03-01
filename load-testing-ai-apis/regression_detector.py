"""
Regression Detection with Welch's t-test and Cohen's d.

From: https://dadops.dev/blog/load-testing-ai-apis/
Code Block 6: "Detecting Degradation Before Your Users Do"

Implements Welch's t-test for unequal variances and a regression
detector that checks both statistical significance and practical
significance (percent change threshold).
"""

import math
import random


def welch_t_test(a, b):
    """Welch's t-test for unequal variances. Returns t-stat and p-value."""
    n_a, n_b = len(a), len(b)
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    t_stat = (mean_a - mean_b) / se if se > 0 else 0

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / den if den > 0 else 1

    # Approximate two-tailed p-value using normal for large df
    p_value = 2 * math.erfc(abs(t_stat) / math.sqrt(2))

    return t_stat, p_value, df


def detect_regression(baseline_ttfts, candidate_ttfts, threshold_pct=10):
    """Compare two sets of TTFT values. Return pass/fail verdict."""
    mean_base = sum(baseline_ttfts) / len(baseline_ttfts)
    mean_cand = sum(candidate_ttfts) / len(candidate_ttfts)
    pct_change = (mean_cand - mean_base) / mean_base * 100

    t_stat, p_value, df = welch_t_test(baseline_ttfts, candidate_ttfts)

    # Cohen's d for effect size
    pooled_std = math.sqrt(
        (sum((x - mean_base)**2 for x in baseline_ttfts) +
         sum((x - mean_cand)**2 for x in candidate_ttfts))
        / (len(baseline_ttfts) + len(candidate_ttfts) - 2)
    )
    cohens_d = abs(mean_cand - mean_base) / pooled_std if pooled_std > 0 else 0

    is_regression = p_value < 0.05 and pct_change > threshold_pct

    print(f"Baseline mean: {mean_base*1000:.1f} ms")
    print(f"Candidate mean: {mean_cand*1000:.1f} ms")
    print(f"Change: {pct_change:+.1f}%")
    print(f"Welch's t={t_stat:.2f}, p={p_value:.4f}, df={df:.1f}")
    print(f"Cohen's d={cohens_d:.2f}")
    print(f"Verdict: {'REGRESSION DETECTED' if is_regression else 'PASS'}")
    return is_regression


if __name__ == "__main__":
    random.seed(42)

    print("=== Regression Detection Demo ===\n")

    # Scenario 1: No regression — similar distributions
    print("--- Scenario 1: No regression (similar distributions) ---")
    baseline = [random.gauss(0.050, 0.005) for _ in range(100)]
    candidate_ok = [random.gauss(0.052, 0.005) for _ in range(100)]
    result1 = detect_regression(baseline, candidate_ok)
    print()

    # Scenario 2: Clear regression — 20% increase
    print("--- Scenario 2: Clear regression (20% increase) ---")
    candidate_bad = [random.gauss(0.060, 0.006) for _ in range(100)]
    result2 = detect_regression(baseline, candidate_bad)
    print()

    # Scenario 3: Borderline — statistically significant but small effect
    print("--- Scenario 3: Borderline (significant but small, <10%) ---")
    candidate_border = [random.gauss(0.054, 0.005) for _ in range(100)]
    result3 = detect_regression(baseline, candidate_border, threshold_pct=10)
    print()

    # Verify Welch's t-test against known values
    print("=== Welch's t-test Verification ===")
    # Simple known case: means clearly different
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [5.0, 6.0, 7.0, 8.0, 9.0]
    t, p, df = welch_t_test(a, b)
    print(f"  Test: a=[1..5], b=[5..9]")
    print(f"  t={t:.4f}, p={p:.6f}, df={df:.1f}")
    print(f"  Expected: t ≈ -4.0, p < 0.01, df ≈ 8.0")
    print(f"  Match: t < 0 and p < 0.01: {t < 0 and p < 0.01}")

    print(f"\n=== Summary ===")
    print(f"  Scenario 1 (no regression): {'PASS' if not result1 else 'FAIL'}")
    print(f"  Scenario 2 (regression):    {'DETECTED' if result2 else 'MISSED'}")
    print(f"  Scenario 3 (borderline):    {'PASS (below threshold)' if not result3 else 'DETECTED'}")
