import numpy as np
from training_basics import init_mlp, forward, cross_entropy, backward_and_update, X, y
from cosine_annealing import cosine_anneal

# --- Warmup + Cosine Decay: The LLM Recipe ---
def warmup_cosine(step, total_steps, warmup_steps=30, lr_max=0.05, lr_min=0.0001):
    """Linear warmup for `warmup_steps`, then cosine decay to lr_min.
    This is the schedule behind GPT-3, LLaMA, and most modern LLMs."""
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps   # linear ramp-up
    # cosine decay over remaining steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

if __name__ == "__main__":
    # --- Show the difference warmup makes ---
    total = 300
    warmup = 30
    for name, sched_fn in [
        ("No warmup (cosine only)", lambda e: cosine_anneal(e, total, lr_max=0.08)),
        ("Warmup + cosine",         lambda e: warmup_cosine(e, total, warmup_steps=warmup, lr_max=0.08)),
    ]:
        params = init_mlp([2, 64, 64, 3])
        losses = []
        for epoch in range(total):
            lr = sched_fn(epoch)
            acts = forward(params, X)
            losses.append(cross_entropy(acts[-1], y))
            backward_and_update(params, acts, y, lr)
        print(f"{name:<28}: final loss = {losses[-1]:.4f}")
