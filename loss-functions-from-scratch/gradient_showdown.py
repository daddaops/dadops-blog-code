import numpy as np

predictions = [0.01, 0.05, 0.20, 0.50, 0.80, 0.95, 0.99]
y = 1  # true label

print(f"{'p':>6} | {'MSE grad':>10} | {'BCE grad':>10} | {'Ratio':>8}")
print("-" * 44)
for p in predictions:
    g_mse = 2 * (p - y) * p * (1 - p)   # MSE gradient (includes sigma' term)
    g_bce = p - y                         # BCE gradient (clean!)
    ratio = abs(g_bce / g_mse) if abs(g_mse) > 1e-10 else float('inf')
    print(f"{p:6.2f} | {g_mse:+10.6f} | {g_bce:+10.4f} | {ratio:7.1f}x")

#      p |   MSE grad |   BCE grad |    Ratio
# --------------------------------------------
#   0.01 |  -0.019602 |    -0.9900 |    50.5x
#   0.05 |  -0.090250 |    -0.9500 |    10.5x
#   0.20 |  -0.256000 |    -0.8000 |     3.1x
#   0.50 |  -0.250000 |    -0.5000 |     2.0x
#   0.80 |  -0.064000 |    -0.2000 |     3.1x
#   0.95 |  -0.004750 |    -0.0500 |    10.5x
#   0.99 |  -0.000198 |    -0.0100 |    50.5x
