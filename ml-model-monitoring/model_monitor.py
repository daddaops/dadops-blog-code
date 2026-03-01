"""
Code Block 5: Complete model monitoring pipeline.

From: https://dadops.dev/blog/ml-model-monitoring/

ModelMonitor ties together drift detection, prediction quality tracking,
feature validation, and CUSUM alerting into a single run() method.

No external dependencies required.
"""

import random
import math


class ModelMonitor:
    """End-to-end model monitoring pipeline."""
    def __init__(self, ref_features, ref_predictions, ref_labels):
        self.ref_features = ref_features
        self.ref_predictions = ref_predictions
        self.ref_pred_mean = sum(ref_predictions) / len(ref_predictions)
        self.cusum = {'psi': 0.0, 'pred_shift': 0.0}
        self.cusum_threshold = 3.0
        self.drift_rate = 0.2

    def _psi(self, ref, prod, bins=10):
        lo = min(min(ref), min(prod))
        hi = max(max(ref), max(prod))
        rng = hi - lo if hi > lo else 1.0
        def hist(data):
            c = [0]*bins
            for v in data:
                c[min(int((v-lo)/rng*bins), bins-1)] += 1
            return [x/len(data)+1e-8 for x in c]
        r, p = hist(ref), hist(prod)
        return sum((pi-ri)*math.log(pi/ri) for ri, pi in zip(r, p))

    def _cusum_update(self, key, value):
        self.cusum[key] = max(0, self.cusum[key] + value - self.drift_rate)
        return self.cusum[key] > self.cusum_threshold

    def run(self, prod_features, prod_predictions, prod_labels=None):
        """Run all monitoring checks. Returns a structured report."""
        report = {'drift': {}, 'quality': {}, 'alerts': []}

        # 1. Feature drift detection
        for feat_name in self.ref_features:
            if feat_name in prod_features:
                psi = self._psi(self.ref_features[feat_name],
                                prod_features[feat_name])
                severity = ("GREEN" if psi < 0.1
                            else "YELLOW" if psi < 0.2 else "RED")
                report['drift'][feat_name] = {'psi': psi, 'severity': severity}

        # 2. Prediction distribution shift
        pred_mean = sum(prod_predictions) / len(prod_predictions)
        pred_shift = abs(pred_mean - self.ref_pred_mean)
        report['quality']['pred_mean_shift'] = pred_shift
        report['quality']['pred_mean'] = pred_mean

        # 3. Quality metrics (if labels available)
        if prod_labels:
            correct = sum(1 for p, l in zip(prod_predictions, prod_labels)
                          if (p > 0.5) == l)
            accuracy = correct / len(prod_labels)
            report['quality']['accuracy'] = accuracy

        # 4. Alerting via CUSUM
        max_psi = max((d['psi'] for d in report['drift'].values()), default=0)
        if self._cusum_update('psi', max_psi):
            report['alerts'].append('CRITICAL: cumulative feature drift detected')
        if self._cusum_update('pred_shift', pred_shift):
            report['alerts'].append('WARNING: prediction distribution shifting')

        return report


if __name__ == "__main__":
    print("=== Model Monitor Pipeline Demo ===\n")

    # Simulate three time periods
    random.seed(42)
    ref_feat = {'age': [random.gauss(35, 10) for _ in range(1000)],
                'spend': [random.gauss(100, 30) for _ in range(1000)]}
    ref_preds = [random.betavariate(2, 5) for _ in range(1000)]
    ref_labels = [1 if random.random() < p else 0 for p in ref_preds]

    monitor = ModelMonitor(ref_feat, ref_preds, ref_labels)

    # Period 1: Stable
    prod1_feat = {'age': [random.gauss(35, 10) for _ in range(200)],
                  'spend': [random.gauss(100, 30) for _ in range(200)]}
    prod1_pred = [random.betavariate(2, 5) for _ in range(200)]
    prod1_lab = [1 if random.random() < p else 0 for p in prod1_pred]
    r1 = monitor.run(prod1_feat, prod1_pred, prod1_lab)

    # Period 2: Gradual drift
    prod2_feat = {'age': [random.gauss(40, 12) for _ in range(200)],
                  'spend': [random.gauss(130, 35) for _ in range(200)]}
    prod2_pred = [min(0.99, p * 1.2 + 0.05) for p in prod1_pred]
    prod2_lab = [1 if random.random() < p else 0 for p in prod1_pred]
    r2 = monitor.run(prod2_feat, prod2_pred, prod2_lab)

    # Period 3: Pipeline break
    prod3_feat = {'age': [0.0] * 200,
                  'spend': [random.gauss(130, 35) for _ in range(200)]}
    prod3_pred = [0.5] * 200  # model confused, outputs uniform
    prod3_lab = [1 if random.random() < 0.25 else 0 for _ in range(200)]
    r3 = monitor.run(prod3_feat, prod3_pred, prod3_lab)

    for label, report in [("Stable", r1), ("Drifted", r2), ("Broken", r3)]:
        drift_summary = {k: f"{v['psi']:.3f} ({v['severity']})"
                         for k, v in report['drift'].items()}
        acc = report['quality'].get('accuracy', 'N/A')
        acc_str = f"{acc:.1%}" if isinstance(acc, float) else acc
        alerts = report['alerts'] if report['alerts'] else ['none']
        print(f"\n[{label}] accuracy={acc_str}")
        print(f"  drift: {drift_summary}")
        print(f"  alerts: {'; '.join(alerts)}")
