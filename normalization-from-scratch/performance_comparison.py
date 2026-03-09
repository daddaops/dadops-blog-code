"""Performance comparison of BatchNorm, LayerNorm, and RMSNorm.

Benchmarks on 3D tensors (batch, sequence, hidden) showing
RMSNorm is faster due to fewer operations (no mean subtraction).
"""
import numpy as np
import time


def batch_norm_3d(x, gamma, beta, eps=1e-5):
    B, S, D = x.shape
    x_flat = x.reshape(-1, D)
    mean = x_flat.mean(axis=0, keepdims=True)
    var  = x_flat.var(axis=0, keepdims=True)
    x_hat = (x_flat - mean) / np.sqrt(var + eps)
    return (gamma * x_hat + beta).reshape(B, S, D)

def layer_norm_3d(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def rms_norm_3d(x, gamma, eps=1e-5):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


np.random.seed(42)
x = np.random.randn(4, 16, 128)   # (batch, seq, hidden)
gamma = np.ones(128)
beta  = np.zeros(128)

bn_out  = batch_norm_3d(x, gamma, beta)
ln_out  = layer_norm_3d(x, gamma, beta)
rms_out = rms_norm_3d(x, gamma)

# Timing: 10,000 iterations each
n = 10000
x_bench = np.random.randn(4, 16, 128)

t0 = time.perf_counter()
for _ in range(n): batch_norm_3d(x_bench, gamma, beta)
t_bn = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(n): layer_norm_3d(x_bench, gamma, beta)
t_ln = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(n): rms_norm_3d(x_bench, gamma)
t_rms = time.perf_counter() - t0

fastest = min(t_bn, t_ln, t_rms)

print(f"Input shape: (4, 16, 128)")
print(f"{'Metric':<20} {'BatchNorm':>12} {'LayerNorm':>12} {'RMSNorm':>12}")
print("-" * 60)
print(f"{'Output mean':<20} {bn_out.mean():>12.4f} {ln_out.mean():>12.4f} {rms_out.mean():>12.4f}")
print(f"{'Output std':<20} {bn_out.std():>12.4f} {ln_out.std():>12.4f} {rms_out.std():>12.4f}")
print(f"{'Parameters':<20} {'256':>12} {'256':>12} {'128':>12}")
print(f"{'Time (10K iters)':<20} {t_bn:>11.1f}s {t_ln:>11.1f}s {t_rms:>11.1f}s")
print(f"{'Relative speed':<20} {t_bn/fastest:>11.1f}x {t_ln/fastest:>11.1f}x {t_rms/fastest:>11.1f}x")
