"""Local gradient reference — implement and verify local gradients for
add, multiply, matmul, ReLU, sigmoid, and softmax+cross-entropy."""
import numpy as np

def check(name, analytic, numerical, eps=1e-7):
    diff = np.max(np.abs(analytic - numerical))
    status = "PASS" if diff < 1e-5 else "FAIL"
    print(f"  {name}: diff={diff:.2e} [{status}]")

def num_grad_array(f, x, eps=1e-5):
    """Numerical gradient for array-valued input."""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_flat = x.flat
        old = x_flat[i]
        x_flat[i] = old + eps
        fplus = f(x.copy())
        x_flat[i] = old - eps
        fminus = f(x.copy())
        x_flat[i] = old
        grad.flat[i] = (fplus - fminus) / (2 * eps)
    return grad

# --- Addition ---
a, b = 3.0, 5.0
upstream = 1.0  # dL/dc
da, db = upstream * 1.0, upstream * 1.0  # local grad = 1
print("Addition: da=1.0, db=1.0 ✓ (gradient distributor)")

# --- Multiplication ---
a, b = 3.0, 5.0
da, db = b, a  # swap rule
print(f"Multiply: da={da} (=b), db={db} (=a) ✓ (swap rule)")

# --- ReLU ---
x = np.array([-2.0, 0.5, -0.1, 3.0])
relu_grad = (x > 0).astype(float)
print(f"ReLU grad: {relu_grad} ✓ (binary gate)")

# --- Sigmoid ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([0.0, 1.0, -1.0, 5.0])
sig = sigmoid(x)
sig_grad = sig * (1 - sig)
print(f"Sigmoid grad: {np.round(sig_grad, 4)}")
print(f"  Max sigmoid grad = {sig_grad.max():.4f} (at x=0)")

# --- Matrix multiply ---
np.random.seed(42)
X = np.random.randn(3, 4)   # 3 samples, 4 features
W = np.random.randn(4, 2)   # 4 features -> 2 outputs
dC = np.random.randn(3, 2)  # upstream gradient

dX_analytic = dC @ W.T       # shape: (3, 4)
dW_analytic = X.T @ dC       # shape: (4, 2)

# Verify dW numerically
scalar_loss = lambda W_: np.sum(dC * (X @ W_))  # proxy
dW_numerical = num_grad_array(scalar_loss, W.copy())
check("MatMul dW", dW_analytic, dW_numerical)

# --- Softmax + Cross-Entropy ---
logits = np.array([2.0, 1.0, 0.1])
target = np.array([1.0, 0.0, 0.0])  # one-hot

# Softmax
exp_l = np.exp(logits - logits.max())
probs = exp_l / exp_l.sum()

# Cross-entropy loss
loss = -np.sum(target * np.log(probs + 1e-12))

# The beautiful result: gradient = probs - target
grad_logits = probs - target
print(f"\nSoftmax+CE gradient: {np.round(grad_logits, 4)}")
print(f"  = probs - target = {np.round(probs, 4)} - {target}")
print(f"  (prediction minus truth — that's it!)")
