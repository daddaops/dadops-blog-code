"""Train a small MLP on a toy dataset using the micrograd engine.

Demonstrates: Neuron, Layer, MLP classes, training loop with
forward pass, MSE loss, gradient zeroing, backward pass, and SGD update.
"""
import random
from micrograd import Value


class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        # w1*x1 + w2*x2 + ... + b
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return activation.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == "__main__":
    random.seed(42)

    model = MLP(3, [4, 4, 1])
    print(f"Number of parameters: {len(model.parameters())}")
    # Layer 1: 4 neurons × (3 weights + 1 bias) = 16
    # Layer 2: 4 neurons × (4 weights + 1 bias) = 20
    # Output:  1 neuron  × (4 weights + 1 bias) = 5
    # Total: 41 parameters

    # Training data: 4 examples with 3 features each
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # desired outputs

    for step in range(50):

        # 1. Forward pass — make predictions
        predictions = [model(x) for x in xs]

        # 2. Loss — mean squared error
        loss = sum((pred - target) ** 2 for pred, target in zip(predictions, ys))

        # 3. Zero all gradients (CRITICAL — backward() uses +=)
        for p in model.parameters():
            p.grad = 0.0

        # 4. Backward pass — compute all gradients
        loss.backward()

        # 5. Update — nudge each parameter in the direction that reduces loss
        learning_rate = 0.05
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if step % 10 == 0:
            print(f"Step {step:3d} | Loss: {loss.data:.6f}")

    # Final predictions
    final_preds = [model(x).data for x in xs]
    print(f"\nFinal predictions: {[f'{p:.4f}' for p in final_preds]}")
    print(f"Targets:           {ys}")
