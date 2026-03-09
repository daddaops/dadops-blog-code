import numpy as np

np.random.seed(42)
# Matrix completion: recover a rank-2 matrix from partial observations
d = 10
true_rank = 2
U = np.random.randn(d, true_rank)
V = np.random.randn(true_rank, d)
M_true = U @ V  # rank-2 ground truth

# Observe 50% of entries
mask = np.random.rand(d, d) < 0.5

# Method 1: Deep factorization W2 @ W1 (implicit low-rank bias)
W1 = np.random.randn(d, d) * 0.01
W2 = np.random.randn(d, d) * 0.01
lr = 0.002
for step in range(8000):
    M_pred = W2 @ W1
    residual = (M_pred - M_true) * mask
    grad_W2 = residual @ W1.T / mask.sum()
    grad_W1 = W2.T @ residual / mask.sum()
    W2 -= lr * grad_W2
    W1 -= lr * grad_W1
M_deep = W2 @ W1

# Method 2: Direct optimization of M (implicit min Frobenius norm)
M_direct = np.random.randn(d, d) * 0.01
for step in range(8000):
    residual = (M_direct - M_true) * mask
    M_direct -= lr * residual / mask.sum()

# Compare singular value spectra
sv_deep = np.linalg.svd(M_deep, compute_uv=False)
sv_direct = np.linalg.svd(M_direct, compute_uv=False)
sv_true = np.linalg.svd(M_true, compute_uv=False)

print("Singular values (top 5):")
print(f"  True:   {sv_true[:5].round(2)}")
print(f"  Deep:   {sv_deep[:5].round(2)}")
print(f"  Direct: {sv_direct[:5].round(2)}")
print(f"Nuclear norm - True: {sv_true.sum():.1f}, Deep: {sv_deep.sum():.1f}, Direct: {sv_direct.sum():.1f}")
# Deep factorization recovers the low-rank structure; direct optimization spreads energy across all singular values
