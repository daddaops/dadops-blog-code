"""Simplified GPTQ: column-by-column quantization with error compensation."""
import numpy as np
from helpers import symmetric_quantize

def gptq_quantize(W, X_cal, bits=4):
    """Simplified GPTQ: column-by-column quantization with
    Hessian-based error compensation."""
    W = W.copy().astype(np.float64)
    n_rows, n_cols = W.shape
    q_max = 2**(bits - 1) - 1

    # Compute Hessian: H = X @ X^T (how columns interact through data)
    H = X_cal @ X_cal.T
    # Regularize for numerical stability
    H += 1e-4 * np.eye(n_cols) * np.mean(np.diag(H))
    # Cholesky-based inverse for speed and stability
    H_inv = np.linalg.inv(H)

    quantized = np.zeros_like(W, dtype=int)
    scales = np.zeros(n_cols)
    errors = []

    # Process column by column
    for col in range(n_cols):
        w_col = W[:, col]

        # Quantize this column (symmetric)
        alpha = np.max(np.abs(w_col)) + 1e-10
        scale = alpha / q_max
        scales[col] = scale
        q_col = np.clip(np.round(w_col / scale), -q_max, q_max)
        quantized[:, col] = q_col.astype(int)

        # Compute the quantization error for this column
        w_hat = scale * q_col
        delta = w_col - w_hat
        errors.append(np.mean(delta ** 2))

        # Compensate: adjust remaining columns using the Hessian
        if col < n_cols - 1:
            h_diag = H_inv[col, col] + 1e-10
            compensation = np.outer(delta, H_inv[col, col+1:] / h_diag)
            W[:, col+1:] += compensation

    return quantized, errors, scales

# Demo: compare RTN vs GPTQ on a small weight matrix
np.random.seed(42)
W_demo = np.random.randn(8, 32).astype(np.float64) * 0.5
X_cal = np.random.randn(32, 64).astype(np.float64) * 0.3

# RTN: just round each weight independently
_, _, W_rtn = symmetric_quantize(W_demo.flatten(), bits=4)
W_rtn = W_rtn.reshape(W_demo.shape)

# GPTQ: round with error compensation
q_gptq, _, gptq_scales = gptq_quantize(W_demo, X_cal, bits=4)
# Dequantize using the scales from GPTQ (computed on compensated weights)
W_gptq = np.zeros_like(W_demo)
for col in range(W_demo.shape[1]):
    W_gptq[:, col] = gptq_scales[col] * q_gptq[:, col]

# Compare OUTPUT error (what actually matters)
Y_ref = W_demo @ X_cal
Y_rtn = W_rtn @ X_cal
Y_gptq = W_gptq @ X_cal

mse_rtn = np.mean((Y_ref - Y_rtn) ** 2)
mse_gptq = np.mean((Y_ref - Y_gptq) ** 2)

print(f"RTN  output MSE: {mse_rtn:.6f}")
print(f"GPTQ output MSE: {mse_gptq:.6f}")
print(f"GPTQ reduction:  {(1 - mse_gptq/mse_rtn)*100:.1f}%")
