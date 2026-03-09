"""Quantization-Aware Training (QAT) vs Post-Training Quantization."""
import numpy as np
from helpers import symmetric_quantize

class FakeQuantize:
    """Fake quantization for QAT: quantize in forward, STE in backward."""

    def __init__(self, bits=4):
        self.bits = bits
        self.q_max = 2**(bits - 1) - 1

    def forward(self, w):
        """Quantize → dequantize (simulates quantization error)."""
        alpha = np.max(np.abs(w)) + 1e-10
        scale = alpha / self.q_max
        q = np.clip(np.round(w / scale), -self.q_max, self.q_max)
        return scale * q

    def backward(self, grad):
        """Straight-Through Estimator: pass gradient unchanged."""
        return grad

# Train a network WITH and WITHOUT QAT
np.random.seed(42)
n_in, n_hidden, n_out = 16, 64, 4
lr = 0.02

# Generate a classification dataset with nonlinear boundaries
X = np.random.randn(500, n_in).astype(np.float32)
targets = ((X[:, 0] * X[:, 1] > 0).astype(int) +
           (X[:, 2] + X[:, 3] > 0.5).astype(int) +
           (X[:, 4] > 0).astype(int))
targets = np.clip(targets, 0, n_out - 1)

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def train_network(use_qat=False, bits=4, epochs=500):
    """Train a 2-layer network, optionally with fake quantization."""
    np.random.seed(123)
    W1 = np.random.randn(n_in, n_hidden) * 0.5
    W2 = np.random.randn(n_hidden, n_out) * 0.5
    fq = FakeQuantize(bits) if use_qat else None

    for epoch in range(epochs):
        W1_eff = fq.forward(W1) if fq else W1
        W2_eff = fq.forward(W2) if fq else W2

        h = np.maximum(0, X @ W1_eff)
        logits = h @ W2_eff
        probs = softmax(logits)

        loss = -np.mean(np.log(probs[np.arange(len(targets)), targets] + 1e-10))

        grad_logits = probs.copy()
        grad_logits[np.arange(len(targets)), targets] -= 1
        grad_logits /= len(targets)

        grad_W2 = h.T @ grad_logits
        grad_h = grad_logits @ W2_eff.T
        grad_h[X @ W1_eff <= 0] = 0
        grad_W1 = X.T @ grad_h

        W1 -= lr * grad_W1
        W2 -= lr * grad_W2

    return W1, W2

# Train both versions
W1_std, W2_std = train_network(use_qat=False)
W1_qat, W2_qat = train_network(use_qat=True, bits=4)

# Now quantize both to 4-bit and evaluate
def evaluate(W1, W2, bits=4):
    _, _, W1_q = symmetric_quantize(W1.flatten(), bits)
    _, _, W2_q = symmetric_quantize(W2.flatten(), bits)
    h = np.maximum(0, X @ W1_q.reshape(W1.shape))
    logits = h @ W2_q.reshape(W2.shape)
    preds = np.argmax(logits, axis=-1)
    return np.mean(preds == targets)

acc_full = evaluate(W1_std, W2_std, bits=32)
acc_ptq  = evaluate(W1_std, W2_std, bits=4)
acc_qat  = evaluate(W1_qat, W2_qat, bits=4)

print(f"Full precision (FP32): {acc_full:.1%}")
print(f"PTQ (INT4):            {acc_ptq:.1%}")
print(f"QAT (INT4):            {acc_qat:.1%}")
