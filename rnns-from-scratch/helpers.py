"""Shared helpers for RNN scripts."""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
