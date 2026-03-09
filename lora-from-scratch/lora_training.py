import numpy as np

def make_spirals(n_points=300, n_classes=3, noise=0.5, rotation=0.0):
    """Generate a spiral dataset with optional rotation."""
    X, y = [], []
    for c in range(n_classes):
        for i in range(n_points // n_classes):
            t = i / (n_points // n_classes) * 4 + c * (2 * np.pi / n_classes)
            r = t / 4
            x1 = r * np.cos(t + rotation) + np.random.randn() * noise * 0.1
            x2 = r * np.sin(t + rotation) + np.random.randn() * noise * 0.1
            X.append([x1, x2])
            y.append(c)
    return np.array(X), np.array(y)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, targets):
    n = len(targets)
    return -np.sum(np.log(probs[range(n), targets] + 1e-9)) / n

class TinyMLP:
    """A 3-layer MLP for classification."""
    def __init__(self, dims):
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        self.activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)  # ReLU
            self.activations.append(x)
        return softmax(x)

    def total_params(self):
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

class LoRAMLP:
    """Wraps a frozen TinyMLP with LoRA adapters on each layer."""
    def __init__(self, base_net, rank=4):
        self.base_weights = [w.copy() for w in base_net.weights]
        self.biases = [b.copy() for b in base_net.biases]
        self.lora_A = []
        self.lora_B = []
        self.rank = rank
        for w in self.base_weights:
            d_in, d_out = w.shape
            A = np.random.randn(d_in, rank) * np.sqrt(2.0 / d_in)
            B = np.zeros((rank, d_out))
            self.lora_A.append(A)
            self.lora_B.append(B)

    def forward(self, x):
        self.activations = [x]
        for i in range(len(self.base_weights)):
            # Frozen base + LoRA path
            x = x @ self.base_weights[i] + x @ self.lora_A[i] @ self.lora_B[i] + self.biases[i]
            if i < len(self.base_weights) - 1:
                x = np.maximum(0, x)
            self.activations.append(x)
        return softmax(x)

    def trainable_params(self):
        return sum(A.size + B.size for A, B in zip(self.lora_A, self.lora_B))

if __name__ == "__main__":
    np.random.seed(42)

    # Phase 1: Train a base network on the original spirals
    X_base, y_base = make_spirals(rotation=0.0)
    base_net = TinyMLP([2, 64, 64, 3])  # 2→64→64→3
    lr = 0.01

    for epoch in range(500):
        probs = base_net.forward(X_base)
        loss = cross_entropy(probs, y_base)

        n = len(y_base)
        grad = probs.copy()
        grad[range(n), y_base] -= 1
        grad /= n

        for i in reversed(range(len(base_net.weights))):
            a = base_net.activations[i]
            dw = a.T @ grad
            db = grad.sum(axis=0)
            if i > 0:
                grad = grad @ base_net.weights[i].T
                grad *= (base_net.activations[i] > 0)
            base_net.weights[i] -= lr * dw
            base_net.biases[i] -= lr * db

    print(f"Base network: {base_net.total_params()} params, final loss: {loss:.4f}")

    # Phase 2: New task — rotated spirals. Base network struggles.
    X_new, y_new = make_spirals(rotation=1.2)
    probs_before = base_net.forward(X_new)
    loss_before = cross_entropy(probs_before, y_new)
    acc_before = np.mean(np.argmax(probs_before, axis=1) == y_new)
    print(f"Base net on new task — loss: {loss_before:.4f}, accuracy: {acc_before:.1%}")

    # Phase 3: LoRA adaptation — freeze base, train only adapters
    lora_net = LoRAMLP(base_net, rank=4)
    lr_lora = 0.02

    for epoch in range(500):
        probs = lora_net.forward(X_new)
        loss = cross_entropy(probs, y_new)

        n = len(y_new)
        grad = probs.copy()
        grad[range(n), y_new] -= 1
        grad /= n

        for i in reversed(range(len(lora_net.base_weights))):
            a = lora_net.activations[i]
            dB = lora_net.lora_A[i].T @ a.T @ grad
            dA = a.T @ grad @ lora_net.lora_B[i].T
            db = grad.sum(axis=0)

            if i > 0:
                W_eff = lora_net.base_weights[i] + lora_net.lora_A[i] @ lora_net.lora_B[i]
                grad = grad @ W_eff.T
                grad *= (lora_net.activations[i] > 0)

            lora_net.lora_A[i] -= lr_lora * dA
            lora_net.lora_B[i] -= lr_lora * dB
            lora_net.biases[i] -= lr_lora * db

    probs_after = lora_net.forward(X_new)
    loss_after = cross_entropy(probs_after, y_new)
    acc_after = np.mean(np.argmax(probs_after, axis=1) == y_new)

    print(f"\nLoRA adaptation results:")
    print(f"  Trainable params: {lora_net.trainable_params()} (vs {base_net.total_params()} full)")
    print(f"  Loss: {loss_before:.4f} → {loss_after:.4f}")
    print(f"  Accuracy: {acc_before:.1%} → {acc_after:.1%}")
