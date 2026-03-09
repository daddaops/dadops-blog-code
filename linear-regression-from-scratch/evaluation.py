import numpy as np

np.random.seed(42)
n = 200

# Scenario 1: Linear data (model is appropriate)
X1 = np.random.randn(n, 3)
y1 = X1 @ [2.0, -1.0, 0.5] + 3.0 + np.random.randn(n) * 0.8

# Scenario 2: Nonlinear data (model is misspecified)
X2 = np.random.randn(n, 1)
y2 = 2 * np.sin(X2[:, 0]) + X2[:, 0] ** 2 + np.random.randn(n) * 0.3

def evaluate(X, y, label):
    X_aug = np.column_stack([np.ones(len(X)), X])
    w = np.linalg.pinv(X_aug) @ y
    y_pred = X_aug @ w
    residuals = y - y_pred

    n, d = X.shape
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - d - 1)
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    print(f"\n--- {label} ---")
    print(f"R²:          {r2:.4f}")
    print(f"Adjusted R²: {adj_r2:.4f}")
    print(f"RMSE:        {rmse:.4f}")
    print(f"MAE:         {mae:.4f}")

    # Residual pattern check
    # Split predictions into 4 bins, check if residual means differ
    bins = np.percentile(y_pred, [25, 50, 75])
    bin_idx = np.digitize(y_pred, bins)
    bin_means = [residuals[bin_idx == i].mean() for i in range(4)]
    pattern = max(bin_means) - min(bin_means)
    if pattern > 0.5:
        print(f"⚠ Residual pattern detected (spread={pattern:.2f}) — model may be misspecified")
    else:
        print(f"✓ Residuals look random (spread={pattern:.2f})")

evaluate(X1, y1, "Linear Data (Good Fit)")
evaluate(X2, y2, "Nonlinear Data (Misspecified Model)")
