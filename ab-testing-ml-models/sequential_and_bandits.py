"""
Sequential testing and multi-armed bandits for A/B testing.

Blog post: https://dadops.dev/blog/ab-testing-ml-models/
Code Blocks 3 & 4: O'Brien-Fleming boundaries, A/A peeking simulation,
Thompson Sampling bandit vs fixed 50/50.

Demonstrates:
- O'Brien-Fleming alpha spending function boundaries
- 10,000-run A/A simulation showing naive peeking inflates false positives
- Thompson Sampling with Beta posteriors vs fixed allocation
- Cumulative regret comparison
"""
import numpy as np
from scipy.stats import norm


def run_sequential_testing():
    """Code Block 3: O'Brien-Fleming boundaries + peeking simulation."""

    def obrien_fleming_bounds(n_looks, alpha=0.05):
        """O'Brien-Fleming spending function boundaries."""
        z_final = norm.ppf(1 - alpha / 2)
        info_fracs = np.linspace(1 / n_looks, 1.0, n_looks)
        boundaries = z_final / np.sqrt(info_fracs)
        return info_fracs, boundaries

    # Show how boundaries change at each interim look
    fracs, bounds = obrien_fleming_bounds(5)
    print("O'Brien-Fleming boundaries (5 interim looks):")
    print(f"{'Look':>6} {'Data%':>8} {'Z-bound':>10} {'p-thresh':>12}")
    print("-" * 40)
    for i, (f, b) in enumerate(zip(fracs, bounds)):
        p_thresh = 2 * (1 - norm.cdf(b))
        print(f"{i+1:>6} {f:>7.0%} {b:>10.3f} {p_thresh:>12.6f}")

    # Simulate: naive peeking vs sequential testing on A/A tests
    np.random.seed(123)
    n_sims, n_total = 10000, 5000
    false_pos_naive, false_pos_seq = 0, 0

    for _ in range(n_sims):
        # A/A test: BOTH groups from the same distribution (no real effect)
        a = np.random.binomial(1, 0.05, n_total)
        b = np.random.binomial(1, 0.05, n_total)

        naive_hit = seq_hit = False
        for look in range(5):
            end = n_total * (look + 1) // 5
            p_a, p_b = a[:end].mean(), b[:end].mean()
            p_pool = (a[:end].sum() + b[:end].sum()) / (2 * end)
            if p_pool == 0 or p_pool == 1:
                continue
            se = np.sqrt(p_pool * (1 - p_pool) * 2 / end)
            z = abs(p_b - p_a) / se

            # Naive: standard alpha at every peek
            if not naive_hit and z > norm.ppf(0.975):
                false_pos_naive += 1
                naive_hit = True
            # Sequential: use O'Brien-Fleming boundary
            if not seq_hit and z > bounds[look]:
                false_pos_seq += 1
                seq_hit = True

    print(f"\nA/A test false positive rate ({n_sims:,} simulations):")
    print(f"  Naive peeking:   {false_pos_naive/n_sims:.1%}  (should be 5%!)")
    print(f"  O'Brien-Fleming: {false_pos_seq/n_sims:.1%}  (correctly controlled)")


def run_thompson_sampling():
    """Code Block 4: Thompson Sampling bandit vs fixed 50/50."""
    np.random.seed(42)
    n_requests = 10000
    true_rates = {"Model A": 0.031, "Model B": 0.038}  # B is better

    # ── Thompson Sampling ──
    alpha_ts = {"Model A": 1, "Model B": 1}  # Beta(1,1) = uniform prior
    beta_ts = {"Model A": 1, "Model B": 1}
    ts_choices = []

    for i in range(n_requests):
        # Sample from each model's posterior
        samples = {m: np.random.beta(alpha_ts[m], beta_ts[m]) for m in true_rates}
        choice = max(samples, key=samples.get)

        # Observe outcome
        reward = np.random.binomial(1, true_rates[choice])
        if reward:
            alpha_ts[choice] += 1
        else:
            beta_ts[choice] += 1
        ts_choices.append(choice)

    # ── Fixed 50/50 split (standard A/B) ──
    np.random.seed(42)
    ab_choices = ["Model A" if i % 2 == 0 else "Model B" for i in range(n_requests)]

    # Compare cumulative regret
    best_rate = max(true_rates.values())
    ts_regret = sum(best_rate - true_rates[c] for c in ts_choices)
    ab_regret = sum(best_rate - true_rates[c] for c in ab_choices)

    ts_b_pct = sum(1 for c in ts_choices if c == "Model B") / n_requests
    print(f"\nAfter {n_requests:,} requests:")
    print(f"  Thompson routed {ts_b_pct:.1%} traffic to Model B (the winner)")
    print(f"  Cumulative regret -- A/B: {ab_regret:.1f} | Thompson: {ts_regret:.1f}")
    print(f"  Regret reduction: {(1 - ts_regret / ab_regret):.0%}")

    # When did Thompson figure it out? (>90% traffic to B)
    window = 200
    for i in range(window, n_requests):
        recent = ts_choices[i - window:i]
        if sum(1 for c in recent if c == "Model B") / window > 0.9:
            print(f"  Thompson routing >90% to B by request {i}")
            break


if __name__ == "__main__":
    run_sequential_testing()
    print("\n" + "=" * 50)
    run_thompson_sampling()
