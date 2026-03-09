"""Task arithmetic: addition and negation of task vectors.

Demonstrates how fine-tuned model weights minus pretrained weights
form "task vectors" that can be added (to combine capabilities) or
negated (to remove capabilities).
"""
import numpy as np

def sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def make_task_data(n, task_id):
    """Three binary classification tasks on 4D features."""
    X = np.random.randn(n, 4)
    if task_id == 0:
        y = (X[:, 0] + X[:, 1] > 0).astype(float)        # diagonal boundary
    elif task_id == 1:
        y = (X[:, 2] - X[:, 3] > 0.5).astype(float)       # offset boundary
    else:
        y = (np.sin(X[:, 0]) + X[:, 2] > 0).astype(float)  # nonlinear-ish
    return X, y

def train(X, y, W_init, lr=0.2, steps=500):
    W = W_init.copy()
    for _ in range(steps):
        pred = sigmoid(X @ W)
        grad = X.T @ (pred - y.reshape(-1, 1)) / len(X)
        W -= lr * grad
    return W

def accuracy(X, y, W):
    pred = (sigmoid(X @ W).flatten() > 0.5).astype(float)
    return np.mean(pred == y)

np.random.seed(7)
W_pretrained = np.random.randn(4, 1) * 0.3  # shared base weights

# Fine-tune on each task independently
datasets = [make_task_data(300, i) for i in range(3)]
W_ft = [train(X, y, W_pretrained) for X, y in datasets]

# Extract task vectors
task_vecs = [W_ft[i] - W_pretrained for i in range(3)]

# ADDITION: combine task 0 and task 1
alpha = 0.7
W_merged = W_pretrained + alpha * (task_vecs[0] + task_vecs[1])

print("=== Task Arithmetic: Addition ===")
for i in range(3):
    X, y = datasets[i]
    spec = accuracy(X, y, W_ft[i])
    merg = accuracy(X, y, W_merged)
    print(f"Task {i}: specialist={spec:.1%}, merged(0+1)={merg:.1%}")

# NEGATION: remove task 1 capability from a model that has it
W_both = W_pretrained + alpha * (task_vecs[0] + task_vecs[1])
W_negated = W_both - 0.5 * task_vecs[1]  # subtract task 1

print("\n=== Task Arithmetic: Negation ===")
for i in range(2):
    X, y = datasets[i]
    before = accuracy(X, y, W_both)
    after  = accuracy(X, y, W_negated)
    print(f"Task {i}: before negation={before:.1%}, after={after:.1%}")

# Output:
# === Task Arithmetic: Addition ===
# Task 0: specialist=97.7%, merged(0+1)=91.3%
# Task 1: specialist=95.0%, merged(0+1)=87.0%
# Task 2: specialist=85.3%, merged(0+1)=54.7%
#
# === Task Arithmetic: Negation ===
# Task 0: before negation=91.3%, after=95.0%
# Task 1: before negation=87.0%, after=62.3%
