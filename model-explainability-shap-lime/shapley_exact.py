"""
Code Block 1: Exact Shapley Values (brute force).

From: https://dadops.dev/blog/model-explainability-shap-lime/

Computes exact Shapley values for a 4-feature loan model by enumerating
all 2^n feature coalitions. Demonstrates the efficiency property:
Shapley values sum exactly to f(x) - E[f(x)].

Dependencies: numpy
"""

import itertools
import math
import numpy as np


# A simple loan model: f(x) = weighted sum + interaction term
def loan_model(income, credit_score, debt_ratio, employment_years):
    score = (0.3 * income + 0.25 * credit_score
             - 0.35 * debt_ratio + 0.1 * employment_years)
    # Add interaction: high income offsets high debt
    score += 0.15 * income * (1 - debt_ratio)
    return score


if __name__ == "__main__":
    print("=== Exact Shapley Values Demo ===\n")

    feature_names = ["income", "credit_score", "debt_ratio", "employment_years"]
    # Applicant to explain (normalized 0-1 scale)
    x = [0.7, 0.8, 0.4, 0.6]
    # Background mean (population average)
    bg = [0.5, 0.5, 0.5, 0.5]

    def f_subset(subset_mask, instance, background):
        """Evaluate model with only subset features; others use background."""
        args = [instance[i] if subset_mask[i] else background[i] for i in range(4)]
        return loan_model(*args)

    n = len(feature_names)
    shapley_values = np.zeros(n)

    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):  # coalition sizes 0..n-1
            for combo in itertools.combinations(others, size):
                mask_without = [j in combo for j in range(n)]
                mask_with = list(mask_without)
                mask_with[i] = True
                marginal = f_subset(mask_with, x, bg) - f_subset(mask_without, x, bg)
                weight = (math.factorial(size)
                          * math.factorial(n - size - 1)
                          / math.factorial(n))
                shapley_values[i] += weight * marginal

    f_x = loan_model(*x)
    f_bg = loan_model(*bg)
    print(f"f(x) = {f_x:.4f},  E[f(x)] = {f_bg:.4f}")
    print(f"f(x) - E[f(x)] = {f_x - f_bg:.4f}")
    print(f"Sum of Shapley values = {shapley_values.sum():.4f}")
    print()
    for name, val in zip(feature_names, shapley_values):
        print(f"  {name:<20s} φ = {val:+.4f}")
