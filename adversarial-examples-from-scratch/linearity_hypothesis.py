"""
The linearity hypothesis: why adversarial attacks work.

Shows that output shift = epsilon * ||w||_1, which grows linearly
with dimension. Also compares adversarial vs random perturbations.

Requires: numpy

From: https://dadops.dev/blog/adversarial-examples-from-scratch/
"""

import numpy as np

if __name__ == "__main__":
    np.random.seed(42)
    dimensions = [10, 50, 100, 500, 1000, 5000]

    print(f"{'Dim':>6s} | {'||w||_1':>8s} | {'shift @ 0.01':>12s} | {'shift @ 0.05':>12s} | {'shift @ 0.10':>12s}")
    print("-" * 65)
    for d in dimensions:
        w = np.random.randn(d) * 0.1          # avg |w_i| ~ 0.08
        l1 = np.sum(np.abs(w))

        shifts = [eps * l1 for eps in [0.01, 0.05, 0.10]]
        print(f"d={d:>4d} | {l1:>8.1f} | {shifts[0]:>12.1f} | {shifts[1]:>12.1f} | {shifts[2]:>12.1f}")

    print("\nThe output shift = epsilon * ||w||_1, which grows with dimension.")
    print("At d=5000 and epsilon=0.01, the shift is ~40x the original output!")

    # Compare adversarial perturbation vs random noise
    d = 1000
    w = np.random.randn(d) * 0.1
    adv_delta = 0.01 * np.sign(w)              # adversarial: aligned with w
    rand_delta = 0.01 * np.sign(np.random.randn(d))  # random direction
    print(f"\nd=1000: adversarial shift = {np.abs(w @ adv_delta):.2f}")
    print(f"d=1000: random noise shift = {np.abs(w @ rand_delta):.2f}")
    print("Adversarial perturbations are special: they align with the weights.")
