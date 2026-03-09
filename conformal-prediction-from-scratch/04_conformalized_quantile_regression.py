"""
Conformalized Quantile Regression (CQR) — adaptive-width prediction intervals.
From: https://dadops.co/blog/conformal-prediction-from-scratch/
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

if __name__ == "__main__":
    # Heteroscedastic data: noise grows with x
    np.random.seed(42)
    X = np.random.uniform(0, 5, 600).reshape(-1, 1)
    noise_scale = 0.3 + 0.5 * X.ravel()  # noise increases with x
    y = np.sin(X.ravel()) * 2 + np.random.randn(600) * noise_scale

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5)
    alpha = 0.10

    # --- Method 1: Basic conformal regression (constant width) ---
    reg = GradientBoostingRegressor(n_estimators=100, random_state=0)
    reg.fit(X_tr, y_tr)
    cal_residuals = np.abs(y_cal - reg.predict(X_cal))
    n = len(cal_residuals)
    q_basic = np.quantile(cal_residuals, np.ceil((1-alpha)*(n+1))/n, method="higher")

    # --- Method 2: CQR (adaptive width) ---
    lo_model = GradientBoostingRegressor(n_estimators=100, loss="quantile",
                                         alpha=alpha/2, random_state=0)
    hi_model = GradientBoostingRegressor(n_estimators=100, loss="quantile",
                                         alpha=1-alpha/2, random_state=0)
    lo_model.fit(X_tr, y_tr)
    hi_model.fit(X_tr, y_tr)

    cal_lo = lo_model.predict(X_cal)
    cal_hi = hi_model.predict(X_cal)
    cqr_scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)
    q_cqr = np.quantile(cqr_scores, np.ceil((1-alpha)*(n+1))/n, method="higher")

    # Evaluate both on test set
    te_pred = reg.predict(X_te)
    te_lo = lo_model.predict(X_te) - q_cqr
    te_hi = hi_model.predict(X_te) + q_cqr

    basic_cov = np.mean((y_te >= te_pred - q_basic) & (y_te <= te_pred + q_basic))
    cqr_cov = np.mean((y_te >= te_lo) & (y_te <= te_hi))
    basic_width = 2 * q_basic
    cqr_width = np.mean(te_hi - te_lo)

    print(f"Basic conformal: coverage={basic_cov:.1%}, avg width={basic_width:.2f}")
    print(f"CQR:             coverage={cqr_cov:.1%}, avg width={cqr_width:.2f}")
