"""
Code Block 3: Real-time cost tracker with attribution and anomaly detection.

From: https://dadops.dev/blog/llm-observability/

Aggregates per-request costs by feature, detects anomalies using z-score
against historical baseline, and generates daily reports.

No external dependencies required.
"""

from collections import defaultdict
from datetime import datetime, timedelta
import statistics


class CostTracker:
    def __init__(self, anomaly_sigma=2.0):
        self.costs = defaultdict(list)   # feature -> [(ts, cost)]
        self.anomaly_sigma = anomaly_sigma

    def record(self, feature, cost_usd, timestamp=None):
        ts = timestamp or datetime.utcnow()
        self.costs[feature].append((ts, cost_usd))

    def feature_cost(self, feature, window=timedelta(hours=24)):
        cutoff = datetime.utcnow() - window
        return sum(c for ts, c in self.costs[feature] if ts > cutoff)

    def detect_anomaly(self, feature, window=timedelta(hours=1)):
        cutoff = datetime.utcnow() - window
        recent = [c for ts, c in self.costs[feature] if ts > cutoff]
        older  = [c for ts, c in self.costs[feature] if ts <= cutoff]

        if len(older) < 10:
            return None                  # not enough baseline data

        baseline_mean = statistics.mean(older)
        baseline_std  = statistics.stdev(older) or 0.001
        current_mean  = statistics.mean(recent) if recent else 0

        z_score = (current_mean - baseline_mean) / baseline_std
        if z_score > self.anomaly_sigma:
            return {
                "feature": feature,
                "current_rate": round(current_mean, 4),
                "baseline_rate": round(baseline_mean, 4),
                "deviation": f"{z_score:.1f}\u03c3",
                "severity": "critical" if z_score > 3.0 else "warning",
            }
        return None

    def daily_report(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        report = {"date": today, "total_cost": 0, "by_feature": {}}
        for feature, entries in self.costs.items():
            day_cost = sum(c for ts, c in entries
                          if ts.strftime("%Y-%m-%d") == today)
            report["by_feature"][feature] = round(day_cost, 2)
            report["total_cost"] += day_cost
        report["total_cost"] = round(report["total_cost"], 2)
        return report


if __name__ == "__main__":
    import json

    print("=== Cost Tracker ===\n")
    tracker = CostTracker()

    # Simulate 24 hours of data with a cost spike
    base = datetime.utcnow() - timedelta(hours=24)

    # Normal baseline: 20 hours of ~$0.01/request
    for i in range(200):
        ts = base + timedelta(hours=i * 20 / 200)
        tracker.record("document-summarizer", 0.01, ts)

    # Cost spike in last hour: $0.03/request
    for i in range(20):
        ts = datetime.utcnow() - timedelta(minutes=60 - i * 3)
        tracker.record("document-summarizer", 0.03, ts)

    # Check anomaly detection
    anomaly = tracker.detect_anomaly("document-summarizer")
    if anomaly:
        print(f"Anomaly detected: {json.dumps(anomaly, indent=2)}")
    else:
        print("No anomaly detected")

    print()

    # Generate daily report
    report = tracker.daily_report()
    print(f"Daily report: {json.dumps(report, indent=2)}")
    print()
    print("Note: daily_report() returns a Python dict, not the")
    print("narrative text shown in the blog's output block.")
