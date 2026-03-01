"""
Expected Gradient Length (EGL): active learning by gradient magnitude.

Scores each unlabeled point by the expected norm of the gradient
if we were to label it. Points that would cause large gradient
updates are the most informative.

Requires: numpy, torch, scikit-learn

From: https://dadops.dev/blog/active-learning-from-scratch/
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification

def expected_gradient_length(model, X_unlabeled, n_classes):
    """Score each point by expected gradient norm."""
    scores = np.zeros(len(X_unlabeled))
    loss_fn = nn.CrossEntropyLoss()

    for i, x in enumerate(X_unlabeled):
        x_tensor = torch.FloatTensor(x).unsqueeze(0)
        probs = torch.softmax(model(x_tensor), dim=1).detach()

        total_grad_norm = 0.0
        for y in range(n_classes):
            # Simulate labeling x with class y
            model.zero_grad()
            output = model(x_tensor)
            loss = loss_fn(output, torch.LongTensor([y]))
            loss.backward()

            grad_norm = sum(p.grad.norm().item() ** 2
                           for p in model.parameters()) ** 0.5
            # Weight by probability of this label
            total_grad_norm += probs[0, y].item() * grad_norm

        scores[i] = total_grad_norm
    return scores


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate 2D binary classification data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               random_state=42)

    # Simple 2-layer network
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    # Train briefly on a small seed set
    seed_idx = []
    for c in [0, 1]:
        class_idx = np.where(y == c)[0][:5]
        seed_idx.extend(class_idx)

    X_seed = torch.FloatTensor(X[seed_idx])
    y_seed = torch.LongTensor(y[seed_idx])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(X_seed)
        loss = loss_fn(out, y_seed)
        loss.backward()
        optimizer.step()

    print(f"Seed training loss: {loss.item():.4f}")

    # Score the full pool
    pool_idx = [i for i in range(len(X)) if i not in seed_idx]
    X_pool = X[pool_idx]

    scores = expected_gradient_length(model, X_pool, n_classes=2)

    # Show top-5 highest EGL scores
    top5 = np.argsort(scores)[-5:][::-1]
    print(f"\nTop-5 EGL scores (pool indices):")
    for rank, idx in enumerate(top5):
        print(f"  #{rank+1}: score={scores[idx]:.4f}, "
              f"point=({X_pool[idx][0]:.2f}, {X_pool[idx][1]:.2f})")

    print(f"\nEGL mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
    print(f"EGL min={np.min(scores):.4f}, max={np.max(scores):.4f}")
