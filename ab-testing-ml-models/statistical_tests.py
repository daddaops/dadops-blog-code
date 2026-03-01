"""
Statistical foundations for A/B testing ML models.

Blog post: https://dadops.dev/blog/ab-testing-ml-models/
Code Blocks 1 & 2: Z-test for proportions, Welch's t-test, sample size calculator.

Demonstrates:
- Two-sample z-test for conversion rate comparison
- Welch's t-test for continuous metrics (revenue per user)
- Sample size calculator with MDE tradeoff table
"""
import numpy as np
from scipy import stats
from scipy.stats import norm


def run_proportion_and_ttest():
    """Code Block 1: Simulate A/B test with z-test and Welch's t-test."""
    np.random.seed(42)
    n_users = 5000  # users per group

    # True conversion rates (unknown to us in practice)
    # Model A (control): 3.1%  |  Model B (treatment): 3.3%
    conv_a = np.random.binomial(1, 0.031, n_users)
    conv_b = np.random.binomial(1, 0.033, n_users)

    # ── Two-sample z-test for proportions ──
    p_a, p_b = conv_a.mean(), conv_b.mean()
    p_pool = (conv_a.sum() + conv_b.sum()) / (2 * n_users)
    se = np.sqrt(p_pool * (1 - p_pool) * (2 / n_users))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    print("=== Proportion Test (Conversion Rate) ===")
    print(f"Model A: {p_a:.4f}  |  Model B: {p_b:.4f}")
    print(f"Lift: {p_b - p_a:+.4f} ({(p_b - p_a) / p_a:+.1%} relative)")
    print(f"Z = {z:.3f}, p-value = {p_value:.4f}")
    print(f"Significant at alpha=0.05? {'Yes' if p_value < 0.05 else 'No'}")

    # ── Welch's t-test for continuous metric (revenue per user) ──
    rev_a = np.where(conv_a, np.random.exponential(45, n_users), 0)
    rev_b = np.where(conv_b, np.random.exponential(48, n_users), 0)

    t_stat, t_pval = stats.ttest_ind(rev_a, rev_b, equal_var=False)
    pooled_std = np.sqrt((rev_a.std()**2 + rev_b.std()**2) / 2)
    cohens_d = (rev_b.mean() - rev_a.mean()) / pooled_std

    se_diff = np.sqrt(rev_a.var() / n_users + rev_b.var() / n_users)
    ci_lo = (rev_b.mean() - rev_a.mean()) - 1.96 * se_diff
    ci_hi = (rev_b.mean() - rev_a.mean()) + 1.96 * se_diff

    print("\n=== Welch's t-test (Revenue per User) ===")
    print(f"Model A: ${rev_a.mean():.2f}  |  Model B: ${rev_b.mean():.2f}")
    print(f"t = {t_stat:.3f}, p = {t_pval:.4f}, Cohen's d = {cohens_d:.3f}")
    print(f"95% CI for difference: (${ci_lo:.2f}, ${ci_hi:.2f})")


def run_sample_size_calculator():
    """Code Block 2: Sample size calculator with tradeoff table."""

    def required_sample_size(p1, p2, alpha=0.05, power=0.80):
        """Minimum users per group to detect a difference p1 vs p2."""
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        num = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
        den = (p2 - p1) ** 2
        return int(np.ceil(num / den))

    # Your scenario: baseline 3.1%, hoping for 3.3%
    n = required_sample_size(0.031, 0.033)
    print(f"\nTo detect 3.1% -> 3.3%: {n:,} users/group ({2*n:,} total)")
    print(f"At 1,000 users/day: {2*n / 1000:.0f} days to complete\n")

    # Tradeoff table: how MDE affects required sample size
    baseline = 0.05  # 5% baseline conversion rate
    print(f"Baseline: {baseline:.0%} conversion rate")
    print(f"{'MDE (abs)':>12} {'Relative':>10} {'N/group':>10} {'Total':>10}")
    print("-" * 46)
    for mde in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        n = required_sample_size(baseline, baseline + mde)
        rel = mde / baseline
        print(f"{mde:>11.1%} {rel:>9.0%} {n:>10,} {2*n:>10,}")


if __name__ == "__main__":
    run_proportion_and_ttest()
    print("\n" + "=" * 50)
    run_sample_size_calculator()
