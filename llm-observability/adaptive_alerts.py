"""
Code Block 7: Adaptive alert system with time-of-week awareness.

From: https://dadops.dev/blog/llm-observability/

Uses historical values at the same hour-of-week to compute dynamic
thresholds (mean + sigma * std). Evaluates 4 metrics: cost_per_min,
latency_p95, quality_score, and error_rate.

No external dependencies required.
"""

from dataclasses import dataclass
from enum import Enum
import statistics


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    name: str
    severity: Severity
    message: str
    value: float
    threshold: float


class AdaptiveAlertSystem:
    def __init__(self):
        self.history = {}   # metric -> [(hour_of_week, value)]

    def record(self, metric, value, hour_of_week):
        self.history.setdefault(metric, []).append(
            (hour_of_week, value))

    def _adaptive_threshold(self, metric, hour_of_week,
                            base_threshold, sigma=2.0):
        same_hour = [v for h, v in self.history.get(metric, [])
                     if h == hour_of_week]
        if len(same_hour) < 4:
            return base_threshold

        mean = statistics.mean(same_hour)
        std  = (statistics.stdev(same_hour)
                if len(same_hour) > 1 else 0)
        return mean + sigma * std

    def evaluate(self, metrics, hour_of_week):
        alerts = []
        rules = [
            ("cost_per_min", 0.50, Severity.CRITICAL,
             "Cost spike: ${value:.2f}/min "
             "(threshold: ${threshold:.2f})"),
            ("latency_p95", 5000, Severity.WARNING,
             "Latency regression: {value:.0f}ms p95 "
             "(threshold: {threshold:.0f}ms)"),
            ("quality_score", 0.70, Severity.WARNING,
             "Quality drop: {value:.2f} "
             "(floor: {threshold:.2f})"),
            ("error_rate", 0.05, Severity.CRITICAL,
             "Error surge: {value:.1%} "
             "(threshold: {threshold:.1%})"),
        ]

        for metric, base, severity, template in rules:
            value = metrics.get(metric)
            if value is None:
                continue

            threshold = self._adaptive_threshold(
                metric, hour_of_week, base)
            self.record(metric, value, hour_of_week)

            is_floor = metric == "quality_score"
            triggered = (value < threshold if is_floor
                         else value > threshold)
            if triggered:
                alerts.append(Alert(
                    name=metric, severity=severity,
                    value=value, threshold=threshold,
                    message=template.format(
                        value=value, threshold=threshold),
                ))
        return alerts


if __name__ == "__main__":
    print("=== Adaptive Alert System ===\n")
    system = AdaptiveAlertSystem()

    # Seed with 4 weeks of normal Monday 9am data (hour_of_week=9)
    for _ in range(4):
        system.record("cost_per_min", 0.25, 9)
        system.record("latency_p95", 2500, 9)
        system.record("quality_score", 0.85, 9)
        system.record("error_rate", 0.02, 9)

    # Normal Monday 9am — should NOT trigger
    normal_metrics = {
        "cost_per_min": 0.30,
        "latency_p95": 2800,
        "quality_score": 0.82,
        "error_rate": 0.025,
    }
    alerts = system.evaluate(normal_metrics, hour_of_week=9)
    print(f"Normal Monday 9am: {len(alerts)} alerts")
    for a in alerts:
        print(f"  {a.severity.value}: {a.message}")

    print()

    # Anomalous Monday 9am — cost spike, quality drop
    anomalous_metrics = {
        "cost_per_min": 1.50,
        "latency_p95": 8000,
        "quality_score": 0.55,
        "error_rate": 0.08,
    }
    alerts = system.evaluate(anomalous_metrics, hour_of_week=9)
    print(f"Anomalous Monday 9am: {len(alerts)} alerts")
    for a in alerts:
        print(f"  {a.severity.value}: {a.message}")

    print()

    # Test quality_score adaptive threshold direction
    # Note: For a floor metric, mean + sigma*std makes threshold HIGHER,
    # meaning it's harder to trigger (more lenient). This is the blog's
    # actual implementation.
    print("Adaptive threshold analysis:")
    print(f"  quality_score history at hour 9: "
          f"{[v for h, v in system.history.get('quality_score', []) if h == 9]}")
    threshold = system._adaptive_threshold("quality_score", 9, 0.70)
    print(f"  Adaptive threshold: {threshold:.4f}")
    print(f"  (mean + 2*std of historical values)")
