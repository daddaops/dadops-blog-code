"""
Segment analysis and complete A/B testing pipeline.

Blog post: https://dadops.dev/blog/ab-testing-ml-models/
Code Blocks 5 & 6: Segment-level analysis with Bonferroni correction,
guardrail checks, and ABTestRunner class with fraud detection simulation.

Demonstrates:
- Segment-level conversion analysis with multiple comparison correction
- Guardrail metrics (latency regression detection)
- Complete ABTestRunner class with sequential boundaries + guardrails
- Fraud detection A/B test that gets blocked by latency guardrail
"""
import hashlib
import numpy as np
from scipy.stats import norm
from collections import defaultdict


def run_segment_analysis():
    """Code Block 5: Segment analysis with Bonferroni + guardrails."""
    np.random.seed(42)

    # Simulated segment-level A/B results
    segments = {
        "Desktop":  {"n": 3000, "rate_a": 0.052, "rate_b": 0.058,
                     "lat_a": 120, "lat_b": 125},
        "Mobile":   {"n": 4000, "rate_a": 0.038, "rate_b": 0.035,
                     "lat_a": 200, "lat_b": 240},
        "Tablet":   {"n": 1000, "rate_a": 0.045, "rate_b": 0.049,
                     "lat_a": 180, "lat_b": 185},
    }

    n_segments = len(segments)
    alpha_bonf = 0.05 / n_segments  # Bonferroni correction

    print(f"Segment analysis (Bonferroni alpha = {alpha_bonf:.4f})")
    print(f"{'Segment':>10} {'A':>8} {'B':>8} {'Lift':>8} {'p-val':>10} {'Sig':>5}")
    print("-" * 53)

    for seg, d in segments.items():
        n = d["n"]
        obs_a = np.random.binomial(n, d["rate_a"]) / n
        obs_b = np.random.binomial(n, d["rate_b"]) / n
        p_pool = (obs_a + obs_b) / 2
        se = np.sqrt(p_pool * (1 - p_pool) * 2 / n) if p_pool > 0 else 1
        z = (obs_b - obs_a) / se
        p_val = 2 * (1 - norm.cdf(abs(z)))
        sig = "Yes" if p_val < alpha_bonf else "No"
        print(f"{seg:>10} {obs_a:>7.1%} {obs_b:>7.1%} {obs_b-obs_a:>+7.1%} "
              f"{p_val:>10.4f} {sig:>5}")

    # ── Guardrail check ──
    print("\n=== Guardrail Check (max 10% latency increase) ===")
    for seg, d in segments.items():
        inc = d["lat_b"] - d["lat_a"]
        pct = inc / d["lat_a"]
        status = "FAIL" if pct > 0.10 else "Pass"
        print(f"  {seg}: {d['lat_a']}ms -> {d['lat_b']}ms "
              f"({pct:+.0%}) [{status}]")


class ABTestRunner:
    """Code Block 6: Complete A/B testing pipeline."""

    def __init__(self, alpha=0.05, n_looks=5):
        self.alpha = alpha
        self.n_looks = n_looks
        self.outcomes = defaultdict(list)
        self.guardrails = {}
        # Pre-compute O'Brien-Fleming boundaries
        fracs = np.linspace(1 / n_looks, 1.0, n_looks)
        z_final = norm.ppf(1 - alpha / 2)
        self.boundaries = z_final / np.sqrt(fracs)

    def assign(self, user_id):
        """Deterministic assignment: same user always gets same group."""
        h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return "treatment" if h % 2 == 0 else "control"

    def record(self, user_id, metric, value):
        group = self.assign(user_id)
        self.outcomes[group].append({"metric": metric, "value": value})

    def add_guardrail(self, metric, max_regression):
        self.guardrails[metric] = max_regression

    def analyze(self, look_index, metric="primary"):
        ctrl = [o["value"] for o in self.outcomes["control"]
                if o["metric"] == metric]
        treat = [o["value"] for o in self.outcomes["treatment"]
                 if o["metric"] == metric]
        mean_c, mean_t = np.mean(ctrl), np.mean(treat)
        se = np.sqrt(np.var(ctrl)/len(ctrl) + np.var(treat)/len(treat))
        z = (mean_t - mean_c) / se if se > 0 else 0

        boundary = self.boundaries[min(look_index, self.n_looks - 1)]
        sig = abs(z) > boundary

        # Check guardrails
        guardrails_ok = True
        for gm, threshold in self.guardrails.items():
            gc = [o["value"] for o in self.outcomes["control"]
                  if o["metric"] == gm]
            gt = [o["value"] for o in self.outcomes["treatment"]
                  if o["metric"] == gm]
            if gc and gt and (np.mean(gt) - np.mean(gc)) > threshold:
                guardrails_ok = False

        return {"z": z, "boundary": boundary, "significant": sig,
                "lift": mean_t - mean_c, "guardrails_pass": guardrails_ok,
                "verdict": ("Ship it!" if sig and guardrails_ok
                            else "Blocked by guardrail" if sig
                            else "Keep testing")}


def run_fraud_simulation():
    """Simulated fraud detection A/B test."""
    runner = ABTestRunner(alpha=0.05, n_looks=5)
    runner.add_guardrail("latency", max_regression=10)

    np.random.seed(42)
    for i in range(5000):
        uid = f"user_{i}"
        group = runner.assign(uid)
        # Fraud v2 catches more fraud (+3pp) but is slower (+12ms avg)
        fraud_rate = 0.12 if group == "control" else 0.155
        base_lat = 50 if group == "control" else 62
        runner.record(uid, "fraud_caught", np.random.binomial(1, fraud_rate))
        runner.record(uid, "latency", np.random.exponential(base_lat))

    result = runner.analyze(look_index=4, metric="fraud_caught")  # Final look
    print(f"\n=== Fraud Detection A/B Test ===")
    print(f"Z = {result['z']:.3f} (boundary: {result['boundary']:.3f})")
    print(f"Significant: {result['significant']}")
    print(f"Guardrails pass: {result['guardrails_pass']}")
    print(f"Verdict: {result['verdict']}")


if __name__ == "__main__":
    run_segment_analysis()
    print("\n" + "=" * 50)
    run_fraud_simulation()
