import numpy as np

def softmax(logits):
    """Numerically stable softmax (subtract-max trick from our softmax post)."""
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)

def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy for one-hot targets."""
    y_pred = np.clip(y_pred, 1e-15, 1.0)
    return -np.sum(y_true * np.log(y_pred))

# 5 classes. True class is index 2.
logits = np.array([2.0, 1.0, 0.5, -1.0, 0.1])
y_true = np.array([0, 0, 1, 0, 0])  # one-hot: class 2

probs = softmax(logits)
loss = categorical_cross_entropy(y_true, probs)

print(f"Probabilities: {np.round(probs, 4)}")
print(f"P(correct class): {probs[2]:.4f}")
print(f"Loss: {loss:.4f}")
# Probabilities: [0.5585 0.2055 0.1246 0.0278 0.0835]
# P(correct class): 0.1246
# Loss: 2.0824

# The gradient of softmax + categorical cross-entropy:
#   dL/dz_k = y_hat_k - y_k   (for ALL k simultaneously)
#
# Predicted probability minus true label. That's it.

def softmax_ce_gradient(probs, y_true):
    """Gradient of (softmax + cross-entropy) w.r.t. logits."""
    return probs - y_true

grad = softmax_ce_gradient(probs, y_true)
print("Gradients per class:")
for k in range(len(grad)):
    marker = " <- true class" if y_true[k] == 1 else ""
    print(f"  class {k}: y_hat={probs[k]:.4f}, y={y_true[k]}, grad={grad[k]:+.4f}{marker}")
# Gradients per class:
#   class 0: y_hat=0.5585, y=0, grad=+0.5585
#   class 1: y_hat=0.2055, y=0, grad=+0.2055
#   class 2: y_hat=0.1246, y=1, grad=-0.8754 <- true class
#   class 3: y_hat=0.0278, y=0, grad=+0.0278
#   class 4: y_hat=0.0835, y=0, grad=+0.0835

def log_softmax(logits):
    """Log-softmax: log(softmax(z)) computed stably."""
    shifted = logits - np.max(logits)
    return shifted - np.log(np.sum(np.exp(shifted)))

def stable_categorical_ce(logits, class_index):
    """Categorical CE using log-softmax (numerically stable)."""
    return -log_softmax(logits)[class_index]

# Same example, stable computation
loss_stable = stable_categorical_ce(logits, class_index=2)
print(f"Stable loss: {loss_stable:.4f}")
# Stable loss: 2.0824  <- same answer, but won't overflow with large logits
