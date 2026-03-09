"""Shared helpers for ResNet scripts."""
import numpy as np


def make_spirals(n_points=200, noise=0.3, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n_points)
    r = np.linspace(0.3, 1.0, n_points)
    x0 = np.column_stack([r * np.cos(t) + rng.randn(n_points) * noise * 0.1,
                          r * np.sin(t) + rng.randn(n_points) * noise * 0.1])
    x1 = np.column_stack([r * np.cos(t + np.pi) + rng.randn(n_points) * noise * 0.1,
                          r * np.sin(t + np.pi) + rng.randn(n_points) * noise * 0.1])
    X = np.vstack([x0, x1])
    y = np.array([0] * n_points + [1] * n_points)
    return X, y


def conv2d_same(x, W):
    """3x3 convolution with same padding. x: (C_in, H, W), W: (C_out, C_in, 3, 3)"""
    C_in, H, W_dim = x.shape
    C_out = W.shape[0]
    x_pad = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((C_out, H, W_dim))
    for co in range(C_out):
        for ci in range(C_in):
            for i in range(H):
                for j in range(W_dim):
                    out[co, i, j] += np.sum(x_pad[ci, i:i+3, j:j+3] * W[co, ci])
    return out


def conv2d(x, W, stride=1, pad=0):
    """General convolution. x: (C_in, H, W), W: (C_out, C_in, kH, kW)"""
    C_in, H, W_dim = x.shape
    C_out, _, kH, kW = W.shape
    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    H_out = (x.shape[1] - kH) // stride + 1
    W_out = (x.shape[2] - kW) // stride + 1
    out = np.zeros((C_out, H_out, W_out))
    for co in range(C_out):
        for ci in range(C_in):
            for i in range(H_out):
                for j in range(W_out):
                    si, sj = i * stride, j * stride
                    out[co, i, j] += np.sum(x[ci, si:si+kH, sj:sj+kW] * W[co, ci])
    return out


def batch_norm(x, gamma, beta, eps=1e-5):
    """Simplified BN over spatial dims. x: (C, H, W)"""
    mean = x.mean(axis=(1, 2), keepdims=True)
    var = x.var(axis=(1, 2), keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma.reshape(-1, 1, 1) * x_norm + beta.reshape(-1, 1, 1)
