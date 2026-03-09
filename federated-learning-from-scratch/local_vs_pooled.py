"""
Local vs Pooled Training

Shows the gap between centralized (pooled) training and local hospital
models — motivating why federated learning is needed.

Blog post: https://dadops.dev/blog/federated-learning-from-scratch/
"""
from setup import *

hospitals, X_all, y_all, true_w = make_hospitals()

w_pooled = train(X_all, y_all)
print(f"Pooled (200 patients): {acc(X_all, y_all, w_pooled):.0%}")

for i, (X, y) in enumerate(hospitals):
    w_local = train(X, y)
    print(f"Hospital {i} (40 patients): {acc(X_all, y_all, w_local):.0%}")
