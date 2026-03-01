"""
Full MLP comparison: train a 4-layer network on spiral data
with 8 different activation functions.

Shows which activations allow deep networks to converge
and which get stuck due to vanishing gradients.

Requires: numpy

From: https://dadops.dev/blog/activation-functions-from-scratch/
"""

import numpy as np

# Generate spiral dataset (2 classes, 200 points each)
np.random.seed(42)
N = 200  # points per class
theta = np.linspace(0, 4 * np.pi, N)
r = np.linspace(0.3, 1, N)
X_a = np.column_stack([r * np.cos(theta) + 0.05 * np.random.randn(N),
                        r * np.sin(theta) + 0.05 * np.random.randn(N)])
X_b = np.column_stack([r * np.cos(theta + np.pi) + 0.05 * np.random.randn(N),
                        r * np.sin(theta + np.pi) + 0.05 * np.random.randn(N)])
X = np.vstack([X_a, X_b])
y = np.hstack([np.zeros(N), np.ones(N)]).reshape(-1, 1)

# MLP: 2 -> 32 -> 32 -> 32 -> 1 (4 layers)
activations = {
    'sigmoid': (lambda x: 1/(1+np.exp(-np.clip(x,-500,500))),
                lambda x: (s:=1/(1+np.exp(-np.clip(x,-500,500))))*(1-s)),
    'tanh':    (np.tanh, lambda x: 1 - np.tanh(x)**2),
    'relu':    (lambda x: np.maximum(0,x), lambda x: (x > 0).astype(float)),
    'lrelu':   (lambda x: np.where(x>0,x,0.01*x),
                lambda x: np.where(x>0,1.0,0.01)),
    'elu':     (lambda x: np.where(x>0,x,np.exp(x)-1),
                lambda x: np.where(x>0,1.0,np.exp(x))),
    'gelu':    (lambda x: 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3))),
                None),  # use numerical grad
    'silu':    (lambda x: x/(1+np.exp(-x)),
                None),  # use numerical grad
    'mish':    (lambda x: x*np.tanh(np.log(1+np.exp(np.clip(x,-20,20)))),
                None),
}

def train_mlp(act_fn, act_grad_fn, epochs=5000, lr=0.5):
    """Train a 4-layer MLP and return loss history + gradient norms."""
    np.random.seed(42)  # same init weights for fair comparison
    dims = [2, 32, 32, 32, 1]
    W = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2/dims[i])
         for i in range(4)]
    b = [np.zeros((1, dims[i+1])) for i in range(4)]

    losses = []
    for epoch in range(epochs):
        # Forward pass — store pre-activations and activations
        a = [X]
        z_list = []
        for i in range(4):
            z = a[-1] @ W[i] + b[i]
            z_list.append(z)
            if i < 3:
                a.append(act_fn(z))
            else:
                a.append(1 / (1 + np.exp(-np.clip(z, -500, 500))))

        # Binary cross-entropy loss
        out = np.clip(a[-1], 1e-7, 1-1e-7)
        loss = -np.mean(y*np.log(out) + (1-y)*np.log(1-out))
        losses.append(loss)

        # Backward pass
        delta = out - y
        for i in range(3, -1, -1):
            dW = a[i].T @ delta / len(X)
            db = delta.mean(axis=0, keepdims=True)
            if i > 0:
                delta = delta @ W[i].T
                if act_grad_fn:
                    delta *= act_grad_fn(z_list[i-1])
                else:
                    eps = 1e-5
                    delta *= (act_fn(z_list[i-1]+eps)-act_fn(z_list[i-1]-eps))/(2*eps)
            W[i] -= lr * dW
            b[i] -= lr * db

    return losses

# Run all activations
results = {}
for name, (fn, grad_fn) in activations.items():
    results[name] = train_mlp(fn, grad_fn)
    final_loss = results[name][-1]
    print(f"{name:8s}: final loss = {final_loss:.4f}")
