import numpy as np

def cross_entropy(p, q):
    """Cross-entropy H(p, q) in bits."""
    p, q = np.array(p, dtype=float), np.array(q, dtype=float)
    q = np.clip(q, 1e-12, 1.0)
    mask = p > 0
    return -np.sum(p[mask] * np.log2(q[mask]))

def binary_cross_entropy(y_true, y_pred):
    """BCE for a single sample (in nats, as used in ML)."""
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_onehot, y_pred):
    """Multi-class cross-entropy for a single sample (in nats)."""
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    return -np.sum(y_onehot * np.log(y_pred))

# --- Show cross-entropy >= entropy ---
p = [0.7, 0.2, 0.1]
q_good = [0.6, 0.3, 0.1]    # close to p
q_bad  = [0.1, 0.1, 0.8]    # very different from p

print(f"H(p)       = {cross_entropy(p, p):.4f} bits (entropy)")
print(f"H(p,q_good)= {cross_entropy(p, q_good):.4f} bits")
print(f"H(p,q_bad) = {cross_entropy(p, q_bad):.4f} bits")

# --- Binary cross-entropy example ---
# True label y=1, model predicts 0.9 (confident and correct)
print(f"\nBCE(y=1, pred=0.9) = {binary_cross_entropy(1, 0.9):.4f}")
# True label y=1, model predicts 0.1 (confident and WRONG)
print(f"BCE(y=1, pred=0.1) = {binary_cross_entropy(1, 0.1):.4f}")

# --- Multi-class example ---
y_true = [0, 1, 0]  # class 1 is correct
y_pred = [0.1, 0.8, 0.1]
print(f"\nCCE = {categorical_cross_entropy(y_true, y_pred):.4f}")
# This equals -log(0.8) — just the negative log of the correct class
