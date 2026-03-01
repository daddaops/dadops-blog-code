"""
Code Block 4: Latency analyzer with percentile breakdown.

From: https://dadops.dev/blog/llm-observability/

Segments latency measurements by model and prompt-length bucket,
computes percentiles (p50, p95, p99), and detects regressions by
comparing the latest window against the previous window.

No external dependencies required.
"""

from collections import defaultdict


class LatencyAnalyzer:
    def __init__(self):
        self.latencies = defaultdict(list)  # (model, bucket) -> [ms]
        self.ttft = defaultdict(list)

    def _bucket(self, tokens):
        if tokens < 500:   return "<500"
        if tokens < 1000:  return "500-1k"
        if tokens < 2000:  return "1k-2k"
        return "2k+"

    def record(self, model, input_tokens, latency_ms, ttft_ms):
        bucket = self._bucket(input_tokens)
        self.latencies[(model, bucket)].append(latency_ms)
        self.ttft[(model, bucket)].append(ttft_ms)

    def percentiles(self, model, bucket=None):
        if bucket:
            data = self.latencies.get((model, bucket), [])
        else:
            data = [v for (m, _), vals in self.latencies.items()
                    if m == model for v in vals]
        if len(data) < 5:
            return None

        data_sorted = sorted(data)
        n = len(data_sorted)
        return {
            "p50": data_sorted[n // 2],
            "p95": data_sorted[int(n * 0.95)],
            "p99": data_sorted[int(n * 0.99)],
            "count": n,
        }

    def detect_regression(self, model, window_size=100):
        all_data = [v for (m, _), vals in self.latencies.items()
                    if m == model for v in vals]
        if len(all_data) < window_size * 2:
            return None

        old = all_data[-window_size * 2 : -window_size]
        new = all_data[-window_size:]
        old_p95 = sorted(old)[int(len(old) * 0.95)]
        new_p95 = sorted(new)[int(len(new) * 0.95)]

        change_pct = (new_p95 - old_p95) / old_p95 * 100
        if change_pct > 20:
            return {"model": model, "old_p95": round(old_p95),
                    "new_p95": round(new_p95),
                    "change": f"+{change_pct:.0f}%"}
        return None


if __name__ == "__main__":
    import random
    import json

    print("=== Latency Analyzer ===\n")
    analyzer = LatencyAnalyzer()
    random.seed(42)

    # Simulate 300 requests with a latency regression in the last 100
    for i in range(200):
        latency = random.gauss(500, 80)
        ttft = latency * 0.15
        tokens = random.choice([200, 600, 1200, 2500])
        analyzer.record("gpt-4o", tokens, latency, ttft)

    # Last 100 requests: latency regression (+40%)
    for i in range(100):
        latency = random.gauss(700, 100)
        ttft = latency * 0.15
        tokens = random.choice([200, 600, 1200, 2500])
        analyzer.record("gpt-4o", tokens, latency, ttft)

    # Percentiles
    pcts = analyzer.percentiles("gpt-4o")
    print(f"Percentiles (all buckets): {json.dumps(pcts, indent=2)}")
    print()

    # Per-bucket percentiles
    for bucket in ["<500", "500-1k", "1k-2k", "2k+"]:
        pcts = analyzer.percentiles("gpt-4o", bucket)
        if pcts:
            print(f"  {bucket}: p50={pcts['p50']:.0f}ms, p95={pcts['p95']:.0f}ms, p99={pcts['p99']:.0f}ms (n={pcts['count']})")

    print()

    # Regression detection
    regression = analyzer.detect_regression("gpt-4o", window_size=100)
    if regression:
        print(f"Regression detected: {json.dumps(regression)}")
    else:
        print("No regression detected")

    # Expected: regression should be detected since last 100 requests have
    # mean 700ms vs previous 200 with mean 500ms
