"""The few-shot adaptation spectrum comparison table.

Displays a summary table of different few-shot adaptation methods
with their accuracy, gradient steps, and notes.
"""

# The few-shot adaptation spectrum
# Each approach trades off adaptation cost vs. performance

results = {
    "Train from scratch": {"acc": 0.20, "grad_steps": 10000,
                           "note": "No prior knowledge"},
    "Prototypical Net":   {"acc": 0.88, "grad_steps": 0,
                           "note": "Nearest centroid in learned space"},
    "MAML (3 steps)":     {"acc": 0.85, "grad_steps": 3,
                           "note": "Adapt from meta-learned init"},
    "FOMAML":             {"acc": 0.83, "grad_steps": 3,
                           "note": "MAML without Hessian"},
    "Reptile":            {"acc": 0.81, "grad_steps": 5,
                           "note": "Move toward task solutions"},
}

print(f"{'Method':<22} {'Acc':>6} {'Steps':>7}  Note")
print("-" * 65)
for method, r in results.items():
    print(f"{method:<22} {r['acc']:>5.0%} {r['grad_steps']:>7d}  {r['note']}")
