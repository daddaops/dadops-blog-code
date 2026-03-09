# Double Descent from Scratch

Verified, runnable code from the DadOps blog post:
[Double Descent from Scratch](https://www.dadops.co/blog/double-descent-from-scratch/)

## Scripts

1. **double_descent_curve.py** — Fits polynomials of increasing degree on a noisy sine wave, showing the classic double descent test error curve with a spike near the interpolation threshold.

2. **three_regimes.py** — Visualizes the three regimes (underfit, critical, overparameterized) by fitting polynomials of degree 5, n, and 5n.

3. **interpolation_peak_regularization.py** — Sweeps the parameter/data ratio and shows how ridge regularization smooths the interpolation peak.

4. **neural_network_double_descent.py** — Demonstrates model-wise double descent with a hand-rolled 2-layer ReLU MLP of increasing width.

5. **minimum_norm_interpolation.py** — Compares minimum-norm (unregularized) solution to optimally-tuned ridge regression in the overparameterized regime.
