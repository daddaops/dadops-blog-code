import numpy as np
from training_basics import init_mlp, forward, cross_entropy, backward_and_update, X, y

# --- Cosine Annealing (standard and with warm restarts) ---
def cosine_anneal(step, total_steps, lr_max=0.05, lr_min=0.0001):
    """Standard cosine decay from lr_max to lr_min."""
    progress = step / max(total_steps - 1, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

def cosine_warm_restarts(step, T_0=100, T_mult=2, lr_max=0.05, lr_min=0.0001):
    """SGDR: cosine with periodic warm restarts (Loshchilov & Hutter 2016).
    T_0: initial cycle length, T_mult: cycle length multiplier."""
    T_cur = T_0
    s = step
    while s >= T_cur:           # find which cycle we're in
        s -= T_cur
        T_cur = int(T_cur * T_mult)
    progress = s / max(T_cur - 1, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))

if __name__ == "__main__":
    # --- Compare: cosine vs cosine with restarts ---
    total = 300
    for name, sched_fn in [
        ("Cosine",          lambda e: cosine_anneal(e, total)),
        ("Cosine+Restarts", lambda e: cosine_warm_restarts(e, T_0=75, T_mult=2)),
    ]:
        params = init_mlp([2, 64, 64, 3])
        losses = []
        for epoch in range(total):
            lr = sched_fn(epoch)
            acts = forward(params, X)
            losses.append(cross_entropy(acts[-1], y))
            backward_and_update(params, acts, y, lr)
        print(f"{name:<20}: final loss = {losses[-1]:.4f}")
