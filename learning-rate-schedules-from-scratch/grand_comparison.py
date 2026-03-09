import numpy as np
from training_basics import init_mlp, forward, cross_entropy, backward_and_update, X, y
from classic_schedules import step_decay, exponential_decay
from cosine_annealing import cosine_anneal, cosine_warm_restarts
from warmup_cosine import warmup_cosine

# --- Grand Comparison: all schedules, same model, same data ---
total_steps = 300

all_schedules = {
    "Constant (0.01)":   lambda s: 0.01,
    "Step Decay":        lambda s: step_decay(s),
    "Exponential":       lambda s: exponential_decay(s),
    "Cosine":            lambda s: cosine_anneal(s, total_steps),
    "Warmup + Cosine":   lambda s: warmup_cosine(s, total_steps, warmup_steps=30),
    "Cosine + Restarts": lambda s: cosine_warm_restarts(s, T_0=75, T_mult=2),
}

print(f"{'Schedule':<20} {'Final Loss':>10} {'Best Loss':>10} {'Converged @':>12}")
print("-" * 56)

for name, sched_fn in all_schedules.items():
    params = init_mlp([2, 64, 64, 3])
    losses = []
    for step in range(total_steps):
        lr = sched_fn(step)
        acts = forward(params, X)
        losses.append(cross_entropy(acts[-1], y))
        backward_and_update(params, acts, y, lr)

    best = min(losses)
    conv_step = next(i for i, l in enumerate(losses) if l < best * 1.05)
    print(f"{name:<20} {losses[-1]:>10.4f} {best:>10.4f} {conv_step:>10}ep")
