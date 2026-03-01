"""
Code Block 4: CUSUM and fixed-threshold alert detectors.

From: https://dadops.dev/blog/ml-model-monitoring/

CUSUMDetector accumulates small deviations to catch gradual drift.
FixedThresholdDetector is a simple comparator for baseline comparison.
Demo shows CUSUM detects gradual drift 8 days earlier than fixed threshold.

No external dependencies required.
"""

import random


class CUSUMDetector:
    """Cumulative Sum detector for gradual drift."""
    def __init__(self, threshold=5.0, drift_rate=0.5):
        self.threshold = threshold
        self.drift_rate = drift_rate
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

    def update(self, value, expected=0.0):
        """Feed a new observation. Returns True if alarm triggered."""
        deviation = value - expected
        self.cusum_pos = max(0, self.cusum_pos + deviation - self.drift_rate)
        self.cusum_neg = max(0, self.cusum_neg - deviation - self.drift_rate)
        return self.cusum_pos > self.threshold or self.cusum_neg > self.threshold

    def reset(self):
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0


class FixedThresholdDetector:
    """Simple fixed-threshold detector for comparison."""
    def __init__(self, threshold=2.0):
        self.threshold = threshold

    def update(self, value, expected=0.0):
        return abs(value - expected) > self.threshold


if __name__ == "__main__":
    print("=== CUSUM vs Fixed Threshold Demo ===\n")

    # Simulate: gradual drift in a monitored metric (e.g., PSI score over days)
    random.seed(42)
    days = 60
    drift_start = 20
    cusum = CUSUMDetector(threshold=4.0, drift_rate=0.3)
    fixed = FixedThresholdDetector(threshold=1.5)

    cusum_alarm_day = None
    fixed_alarm_day = None

    for day in range(days):
        # Gradual drift starting at day 20: PSI increases by 0.02 per day
        if day < drift_start:
            psi = random.gauss(0.05, 0.3)  # normal noise around baseline
        else:
            psi = random.gauss(0.05 + 0.08 * (day - drift_start), 0.3)

        if cusum.update(psi) and cusum_alarm_day is None:
            cusum_alarm_day = day
        if fixed.update(psi) and fixed_alarm_day is None:
            fixed_alarm_day = day

    print(f"Drift starts at day {drift_start}")
    print(f"CUSUM alarm:           day {cusum_alarm_day}")
    print(f"Fixed threshold alarm: day {fixed_alarm_day}")
    print(f"CUSUM advantage:       {(fixed_alarm_day or days) - (cusum_alarm_day or days)} days earlier")
