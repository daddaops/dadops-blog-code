import numpy as np
from training_basics import init_mlp, forward, cross_entropy, backward_and_update, X, y

# --- Classic LR Schedule Functions ---
def step_decay(epoch, lr0=0.05, drop=0.5, every=100):
    """Drop LR by `drop` every `every` epochs."""
    return lr0 * (drop ** (epoch // every))

def exponential_decay(epoch, lr0=0.05, gamma=0.99):
    """Smooth exponential decay: LR = lr0 * gamma^epoch."""
    return lr0 * (gamma ** epoch)

def inverse_sqrt_decay(epoch, lr0=0.05, warmup=1):
    """LR = lr0 / sqrt(epoch). Used in original Transformer."""
    return lr0 / np.sqrt(max(epoch, warmup))

if __name__ == "__main__":
    # --- Train and compare ---
    schedules = [
        ("Constant",     lambda e: 0.01),
        ("Step Decay",   step_decay),
        ("Exponential",  exponential_decay),
        ("Inv Sqrt",     inverse_sqrt_decay),
    ]

    for name, sched_fn in schedules:
        params = init_mlp([2, 64, 64, 3])
        losses = []
        for epoch in range(300):
            lr = sched_fn(epoch)
            acts = forward(params, X)
            losses.append(cross_entropy(acts[-1], y))
            backward_and_update(params, acts, y, lr)
        print(f"{name:<14}: final loss = {losses[-1]:.4f}")
