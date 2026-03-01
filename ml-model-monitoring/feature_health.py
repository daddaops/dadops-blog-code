"""
Code Block 3: Feature quality monitoring.

From: https://dadops.dev/blog/ml-model-monitoring/

FeatureHealthMonitor builds statistical profiles from reference data
and checks production data for null rate spikes, mean shifts, range
violations, and unexpected categorical values.

No external dependencies required.
"""

import random
import math


class FeatureHealthMonitor:
    def __init__(self, reference_data):
        """Build statistical profile from training-time reference data."""
        self.profiles = {}
        for feature_name, values in reference_data.items():
            clean = [v for v in values if v is not None]
            null_rate = 1 - len(clean) / len(values) if values else 0
            if clean and isinstance(clean[0], (int, float)):
                mean = sum(clean) / len(clean)
                std = math.sqrt(sum((v - mean)**2 for v in clean) / len(clean))
                self.profiles[feature_name] = {
                    'type': 'numeric', 'mean': mean, 'std': max(std, 1e-8),
                    'min': min(clean), 'max': max(clean),
                    'null_rate': null_rate,
                }
            else:
                card = len(set(clean))
                self.profiles[feature_name] = {
                    'type': 'categorical', 'cardinality': card,
                    'values': set(clean), 'null_rate': null_rate,
                }

    def check(self, production_data):
        """Check production data against reference profiles."""
        report = {}
        for feature_name, profile in self.profiles.items():
            values = production_data.get(feature_name, [])
            issues = []
            # Null rate check
            null_count = sum(1 for v in values if v is None)
            null_rate = null_count / len(values) if values else 0
            if null_rate > profile['null_rate'] * 10 + 0.01:
                issues.append(f"null rate {null_rate:.1%} vs ref {profile['null_rate']:.1%}")

            clean = [v for v in values if v is not None]
            if profile['type'] == 'numeric' and clean:
                mean = sum(clean) / len(clean)
                z_score = abs(mean - profile['mean']) / profile['std']
                if z_score > 3:
                    issues.append(f"mean shifted {z_score:.1f}σ")
                if min(clean) < profile['min'] - 3 * profile['std']:
                    issues.append(f"min {min(clean):.1f} below range")
                if max(clean) > profile['max'] + 3 * profile['std']:
                    issues.append(f"max {max(clean):.1f} above range")
            elif profile['type'] == 'categorical' and clean:
                new_vals = set(clean) - profile['values']
                if len(new_vals) > profile['cardinality'] * 0.1:
                    issues.append(f"{len(new_vals)} new categories")

            status = "RED" if issues else "GREEN"
            report[feature_name] = {'status': status, 'issues': issues}
        return report


if __name__ == "__main__":
    print("=== Feature Health Monitor Demo ===\n")

    # Simulate: training data profile vs broken production pipeline
    random.seed(42)
    reference = {
        'age': [random.gauss(35, 10) for _ in range(1000)],
        'income': [random.gauss(60000, 5000) for _ in range(1000)],
        'plan': [random.choice(['basic', 'premium', 'enterprise']) for _ in range(1000)],
    }

    # Production: income pipeline broke (sending stale zeros), plan field case-changed
    production = {
        'age': [random.gauss(36, 10) for _ in range(500)],
        'income': [0.0] * 350 + [random.gauss(60000, 5000) for _ in range(150)],
        'plan': [random.choice(['Basic', 'Premium', 'Enterprise']) for _ in range(500)],
    }

    monitor = FeatureHealthMonitor(reference)
    report = monitor.check(production)
    for feature, result in report.items():
        status_icon = "\u2713" if result['status'] == "GREEN" else "\u2717"
        print(f"  {status_icon} {feature:<10s} [{result['status']}] "
              + (', '.join(result['issues']) if result['issues'] else 'OK'))
