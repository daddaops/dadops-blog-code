import numpy as np
from mse_and_sigmoid import sigmoid, sigmoid_derivative, mse_grad_wrt_logit

# Likelihood = p^y * (1-p)^(1-y)
# When y=1: Likelihood = p      (we want p to be high)
# When y=0: Likelihood = 1-p    (we want p to be low)

# Take the log (turns products into sums, easier to optimize):
# log-likelihood = y*log(p) + (1-y)*log(1-p)

# Negate it (because we minimize loss, not maximize likelihood):
# BCE = -[y*log(p) + (1-y)*log(1-p)]

# When y=1: BCE = -log(p)
for p in [0.99, 0.9, 0.5, 0.1, 0.01]:
    print(f"  p={p:.2f} -> loss = {-np.log(p):.4f}")
#   p=0.99 -> loss = 0.0101     <- confident and RIGHT: tiny penalty
#   p=0.90 -> loss = 0.1054     <- mostly right: small penalty
#   p=0.50 -> loss = 0.6931     <- coin flip: moderate penalty
#   p=0.10 -> loss = 2.3026     <- mostly wrong: large penalty
#   p=0.01 -> loss = 4.6052     <- confident and WRONG: massive penalty

def binary_cross_entropy(p, y):
    """Binary cross-entropy loss for a single example."""
    epsilon = 1e-15  # numerical safety: avoid log(0)
    p = np.clip(p, epsilon, 1 - epsilon)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# Same scenario as before: p=0.011, y=1
loss = binary_cross_entropy(0.011, 1)
print(f"BCE loss: {loss:.4f}")
# BCE loss: 4.5099
# Much higher than MSE's 0.978 — cross-entropy screams louder when wrong

# dBCE/dz via chain rule:
#   dBCE/dp = -y/p + (1-y)/(1-p)
#   dp/dz   = p(1-p)            <- the sigmoid derivative
#
# Multiply them:
#   dBCE/dz = [-y/p + (1-y)/(1-p)] * p(1-p)
#           = -y(1-p) + (1-y)p
#           = -y + yp + p - yp
#           = p - y
#
# THE SIGMOID DERIVATIVE CANCELS OUT.

def bce_grad_wrt_logit(p, y):
    """Gradient of BCE w.r.t. the pre-sigmoid logit z."""
    return p - y  # That's it. No sigmoid derivative anywhere.

# Model predicts p=0.011 for a positive example (y=1):
grad_bce = bce_grad_wrt_logit(0.011, 1)
grad_mse = mse_grad_wrt_logit(0.011, 1)

print(f"BCE gradient: {grad_bce:.4f}")
print(f"MSE gradient: {grad_mse:.4f}")
# BCE gradient: -0.9890     <- STRONG push! Nearly -1
# MSE gradient: -0.0215     <- Feeble whisper
