import numpy as np

def hard_argmax(scores):
    """Non-differentiable: returns one-hot vector."""
    idx = np.argmax(scores)
    one_hot = np.zeros_like(scores)
    one_hot[idx] = 1.0
    return one_hot  # gradient is zero everywhere

def soft_argmax(scores, tau=1.0):
    """Differentiable relaxation with temperature."""
    shifted = scores - np.max(scores)  # numerical stability
    exps = np.exp(shifted / tau)
    return exps / np.sum(exps)  # smooth, gradient flows

scores = np.array([2.0, 5.0, 1.0, 3.0])

print("Hard argmax:", hard_argmax(scores))
# [0. 1. 0. 0.] -- no gradient information

print("Soft (tau=1.0):", np.round(soft_argmax(scores, 1.0), 3))
# [0.041 0.831 0.015 0.112] -- smooth, peaked at index 1

print("Soft (tau=0.1):", np.round(soft_argmax(scores, 0.1), 3))
# [0.000 1.000 0.000 0.000] -- nearly hard, but still differentiable

print("Soft (tau=5.0):", np.round(soft_argmax(scores, 5.0), 3))
# [0.206 0.375 0.168 0.251] -- very smooth, gradients everywhere
