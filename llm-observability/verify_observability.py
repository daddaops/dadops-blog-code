"""
Verification suite for all LLM Observability code blocks.

Tests all non-async, non-API logic from the blog post:
- Cost calculations and pricing
- Latency bucketing and percentiles
- Quality monitoring with cosine similarity
- Trace builder arithmetic
- Adaptive alert thresholds and triggering
- Blog claim verification (specific numbers)

No external dependencies required.
"""

import statistics
from datetime import datetime, timedelta

# Import all modules
from llm_call_logger import MODEL_PRICING, LLMCallLog
from cost_tracker import CostTracker
from latency_analyzer import LatencyAnalyzer
from quality_monitor import QualityMonitor
from trace_builder import Trace, Span
from adaptive_alerts import AdaptiveAlertSystem, Severity


def test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if test(name, condition, detail):
            passed += 1
        else:
            failed += 1

    # ═══════════════════════════════════════════
    # 1. MODEL_PRICING verification
    # ═══════════════════════════════════════════
    print("=== 1. Model Pricing ===")
    check("gpt-4o pricing",
          MODEL_PRICING["gpt-4o"] == (2.50, 10.00))
    check("gpt-4o-mini pricing",
          MODEL_PRICING["gpt-4o-mini"] == (0.15, 0.60))
    check("claude-sonnet-4 pricing",
          MODEL_PRICING["claude-sonnet-4"] == (3.00, 15.00))

    # ═══════════════════════════════════════════
    # 2. Blog JSON example cost verification
    # ═══════════════════════════════════════════
    print("\n=== 2. Blog JSON Example Cost ===")
    inp_price, out_price = MODEL_PRICING["gpt-4o"]
    cost = (2847 * inp_price + 312 * out_price) / 1_000_000
    rounded = round(cost, 6)
    check("Cost calculation raw",
          abs(cost - 0.0102375) < 1e-10,
          f"raw={cost}")
    check("Cost rounded to 6 decimals",
          rounded == 0.010238,
          f"rounded={rounded}, blog claims 0.010237")
    # Blog says 0.010237 but Python round(0.0102375, 6) = 0.010238
    # This is a minor rounding discrepancy

    # TTFT verification: 275.1 / 1834.2 ≈ 0.15
    ttft_ratio = 275.1 / 1834.2
    check("TTFT is ~15% of latency",
          abs(ttft_ratio - 0.15) < 0.001,
          f"ratio={ttft_ratio:.5f}")

    # ═══════════════════════════════════════════
    # 3. CostTracker
    # ═══════════════════════════════════════════
    print("\n=== 3. CostTracker ===")
    tracker = CostTracker(anomaly_sigma=2.0)

    # Not enough data → None
    tracker.record("test", 0.01, datetime.utcnow() - timedelta(hours=2))
    check("Anomaly None with <10 baseline points",
          tracker.detect_anomaly("test") is None)

    # Add enough baseline data
    base = datetime.utcnow() - timedelta(hours=5)
    for i in range(15):
        tracker.record("test", 0.01,
                       base + timedelta(minutes=i * 10))
    # Add anomalous recent data
    for i in range(5):
        tracker.record("test", 0.10,
                       datetime.utcnow() - timedelta(minutes=30 - i * 5))
    anomaly = tracker.detect_anomaly("test")
    check("Anomaly detected with spike", anomaly is not None)
    if anomaly:
        check("Anomaly has correct keys",
              all(k in anomaly for k in
                  ["feature", "current_rate", "baseline_rate",
                   "deviation", "severity"]))
        check("Severity is warning or critical",
              anomaly["severity"] in ("warning", "critical"))
        check("Deviation has sigma symbol",
              "\u03c3" in anomaly["deviation"])

    # Daily report format
    report = tracker.daily_report()
    check("Report has date key", "date" in report)
    check("Report has total_cost key", "total_cost" in report)
    check("Report has by_feature key", "by_feature" in report)

    # ═══════════════════════════════════════════
    # 4. LatencyAnalyzer
    # ═══════════════════════════════════════════
    print("\n=== 4. LatencyAnalyzer ===")
    analyzer = LatencyAnalyzer()

    # Test bucketing
    check("Bucket <500", analyzer._bucket(200) == "<500")
    check("Bucket 500-1k", analyzer._bucket(600) == "500-1k")
    check("Bucket 1k-2k", analyzer._bucket(1500) == "1k-2k")
    check("Bucket 2k+", analyzer._bucket(3000) == "2k+")
    check("Bucket boundary 500", analyzer._bucket(500) == "500-1k")
    check("Bucket boundary 1000", analyzer._bucket(1000) == "1k-2k")
    check("Bucket boundary 2000", analyzer._bucket(2000) == "2k+")

    # Not enough data → None
    for i in range(3):
        analyzer.record("gpt-4o", 200, 100 + i * 10, 15)
    check("Percentiles None with <5 points",
          analyzer.percentiles("gpt-4o") is None)

    # Add more data
    for i in range(10):
        analyzer.record("gpt-4o", 200, 100 + i * 50, 15 + i * 5)
    pcts = analyzer.percentiles("gpt-4o")
    check("Percentiles returns dict with 5+ points", pcts is not None)
    if pcts:
        check("Percentiles has p50/p95/p99/count",
              all(k in pcts for k in ["p50", "p95", "p99", "count"]))
        check("p50 <= p95 <= p99",
              pcts["p50"] <= pcts["p95"] <= pcts["p99"])

    # Regression detection needs 2*window_size points
    check("Regression None with insufficient data",
          analyzer.detect_regression("gpt-4o", window_size=100) is None)

    # ═══════════════════════════════════════════
    # 5. QualityMonitor
    # ═══════════════════════════════════════════
    print("\n=== 5. QualityMonitor ===")

    # Cosine similarity
    check("Cosine sim identical vectors",
          abs(QualityMonitor._cosine_sim([1, 0], [1, 0]) - 1.0) < 0.001)
    check("Cosine sim orthogonal vectors",
          abs(QualityMonitor._cosine_sim([1, 0], [0, 1]) - 0.0) < 0.001)
    check("Cosine sim opposite vectors",
          abs(QualityMonitor._cosine_sim([1, 0], [-1, 0]) - (-1.0)) < 0.001)

    # Health score weights
    monitor = QualityMonitor(golden_embeddings=[[1, 0, 0]])
    check("Empty health score", monitor.health_score() == {})

    # Need >200 entries for length stability
    for _ in range(201):
        monitor.record_response("word " * 100, embedding=[0.9, 0.1, 0.1])
        monitor.record_judge_score(4.0)
    scores = monitor.health_score()
    check("Health score has overall key", "overall" in scores)
    check("Health score has length_stability",
          "length_stability" in scores)
    check("Health score has semantic_similarity",
          "semantic_similarity" in scores)
    check("Health score has judge_score",
          "judge_score" in scores)
    check("Judge score normalized (4.0/5.0 = 0.8)",
          abs(scores["judge_score"] - 0.8) < 0.01,
          f"actual={scores['judge_score']:.4f}")

    # Verify weights
    w = {"length_stability": 0.2,
         "semantic_similarity": 0.4,
         "judge_score": 0.4}
    total_w = sum(w[k] for k in scores if k != "overall")
    expected_overall = sum(scores[k] * w[k]
                          for k in scores if k != "overall") / total_w
    check("Overall matches weighted average",
          abs(scores["overall"] - round(expected_overall, 3)) < 0.001,
          f"overall={scores['overall']}, expected={round(expected_overall, 3)}")

    # ═══════════════════════════════════════════
    # 6. Trace Builder
    # ═══════════════════════════════════════════
    print("\n=== 6. Trace Builder ===")

    # Verify blog's example trace numbers
    total_tokens = (12+0) + (0+0) + (2100+50) + (3400+350) + (160+10)
    check("Blog trace total tokens = 6082",
          total_tokens == 6082, f"actual={total_tokens}")

    total_cost = 0.0000 + 0.0000 + 0.0003 + 0.0042 + 0.0002
    check("Blog trace total cost = $0.0047",
          abs(total_cost - 0.0047) < 0.00001, f"actual={total_cost}")

    total_ms = 45 + 120 + 340 + 1755 + 80
    check("Blog trace total ms = 2340",
          total_ms == 2340, f"actual={total_ms}")

    gen_cost_pct = 0.0042 / 0.0047 * 100
    check("Generation is 89% of cost",
          abs(gen_cost_pct - 89.4) < 0.1,
          f"actual={gen_cost_pct:.1f}%")

    gen_lat_pct = 1755 / 2340 * 100
    check("Generation is 75% of latency",
          abs(gen_lat_pct - 75.0) < 0.1,
          f"actual={gen_lat_pct:.1f}%")

    # Test Trace summary format
    trace = Trace(trace_id="test123")
    import time
    with trace.span("step1") as s:
        time.sleep(0.001)
        s.tokens_in = 10; s.tokens_out = 5; s.cost_usd = 0.001
    with trace.span("step2") as s:
        time.sleep(0.001)
        s.tokens_in = 20; s.tokens_out = 10; s.cost_usd = 0.002
    summary = trace.summary()
    check("Summary starts with Trace ID",
          summary.startswith("Trace test123"))
    check("Summary has tree chars",
          "\u251c\u2500" in summary and "\u2514\u2500" in summary)
    check("Summary has 3 lines (header + 2 spans)",
          len(summary.split("\n")) == 3)

    # ═══════════════════════════════════════════
    # 7. AdaptiveAlertSystem
    # ═══════════════════════════════════════════
    print("\n=== 7. AdaptiveAlertSystem ===")
    system = AdaptiveAlertSystem()

    # Falls back to base threshold with <4 data points
    threshold = system._adaptive_threshold(
        "cost_per_min", 9, 0.50)
    check("Base threshold with <4 points", threshold == 0.50)

    # Seed with 4 identical values
    for _ in range(4):
        system.record("test_metric", 1.0, 9)
    threshold = system._adaptive_threshold("test_metric", 9, 0.50)
    check("Adaptive threshold with identical values = mean",
          abs(threshold - 1.0) < 0.001,
          f"threshold={threshold} (std=0, so mean+2*0=mean)")

    # Seed with varying values
    system2 = AdaptiveAlertSystem()
    vals = [1.0, 2.0, 3.0, 4.0]
    for v in vals:
        system2.record("test2", v, 9)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals)
    expected = mean + 2.0 * std
    threshold = system2._adaptive_threshold("test2", 9, 0.50)
    check("Adaptive threshold = mean + 2*std",
          abs(threshold - expected) < 0.001,
          f"threshold={threshold:.4f}, expected={expected:.4f}")

    # quality_score is a floor metric
    system3 = AdaptiveAlertSystem()
    for _ in range(4):
        system3.record("quality_score", 0.85, 9)
    alerts = system3.evaluate({"quality_score": 0.50}, hour_of_week=9)
    check("Quality drop triggers alert",
          any(a.name == "quality_score" for a in alerts))

    # quality_score above threshold should NOT trigger
    system4 = AdaptiveAlertSystem()
    for _ in range(4):
        system4.record("quality_score", 0.85, 9)
    alerts = system4.evaluate({"quality_score": 0.90}, hour_of_week=9)
    check("Quality above threshold no alert",
          not any(a.name == "quality_score" for a in alerts))

    # cost_per_min above threshold triggers
    system5 = AdaptiveAlertSystem()
    for _ in range(4):
        system5.record("cost_per_min", 0.25, 9)
    alerts = system5.evaluate({"cost_per_min": 2.00}, hour_of_week=9)
    check("Cost spike triggers alert",
          any(a.name == "cost_per_min" for a in alerts))

    # error_rate severity is CRITICAL
    system6 = AdaptiveAlertSystem()
    alerts = system6.evaluate({"error_rate": 0.10}, hour_of_week=9)
    check("Error rate alert is CRITICAL",
          any(a.severity == Severity.CRITICAL
              for a in alerts if a.name == "error_rate"))

    # ═══════════════════════════════════════════
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, "
          f"{passed + failed} total")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"WARNING: {failed} test(s) failed")


if __name__ == "__main__":
    main()
