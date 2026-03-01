"""
Code Block 2: Prediction quality monitoring.

From: https://dadops.dev/blog/ml-model-monitoring/

expected_calibration_error() — ECE: weighted |accuracy - confidence| per bin.
prediction_distribution_shift() — L1 distance between prediction histograms.

No external dependencies required.
"""

import random


def expected_calibration_error(predictions, labels, n_bins=10):
    """ECE: weighted average of |accuracy - confidence| per bin."""
    bins = [[] for _ in range(n_bins)]
    for pred, label in zip(predictions, labels):
        idx = min(int(pred * n_bins), n_bins - 1)
        bins[idx].append((pred, label))

    ece = 0.0
    total = len(predictions)
    for bin_data in bins:
        if not bin_data:
            continue
        avg_confidence = sum(p for p, _ in bin_data) / len(bin_data)
        avg_accuracy = sum(l for _, l in bin_data) / len(bin_data)
        ece += len(bin_data) / total * abs(avg_accuracy - avg_confidence)
    return ece


def prediction_distribution_shift(ref_preds, prod_preds, n_bins=10):
    """Compare prediction score distributions between reference and production."""
    def to_hist(preds):
        counts = [0] * n_bins
        for p in preds:
            idx = min(int(p * n_bins), n_bins - 1)
            counts[idx] += 1
        return [c / len(preds) for c in counts]

    ref_hist = to_hist(ref_preds)
    prod_hist = to_hist(prod_preds)
    # L1 distance between histograms
    shift = sum(abs(r - p) for r, p in zip(ref_hist, prod_hist)) / 2
    return shift


if __name__ == "__main__":
    print("=== Prediction Quality Demo ===\n")

    # Simulate: well-calibrated model at launch vs drifted model at month 6
    random.seed(42)
    n = 500

    # At launch: well-calibrated model
    launch_preds = [random.betavariate(2, 5) for _ in range(n)]
    launch_labels = [1 if random.random() < p else 0 for p in launch_preds]

    # At month 6: model is overconfident (predictions shifted toward extremes)
    month6_preds = [min(0.99, max(0.01, p * 1.4 + 0.1)) for p in launch_preds]
    month6_labels = [1 if random.random() < p_orig else 0
                     for p_orig in launch_preds]  # reality unchanged

    print("At launch:")
    print(f"  ECE = {expected_calibration_error(launch_preds, launch_labels):.4f}")
    print(f"  Prediction shift = {prediction_distribution_shift(launch_preds, launch_preds):.4f}")

    print("At month 6 (overconfident):")
    print(f"  ECE = {expected_calibration_error(month6_preds, month6_labels):.4f}")
    print(f"  Prediction shift = {prediction_distribution_shift(launch_preds, month6_preds):.4f}")
