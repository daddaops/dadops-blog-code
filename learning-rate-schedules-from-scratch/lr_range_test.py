import numpy as np
from training_basics import init_mlp, forward, cross_entropy, backward_and_update, X, y

# --- LR Range Test ---
def lr_range_test(X, y, lr_start=1e-6, lr_end=1.0, num_steps=200):
    """Exponentially increase LR each step, record loss.
    Returns (lrs, losses, smoothed_losses)."""
    params = init_mlp([2, 64, 64, 3])
    mult = (lr_end / lr_start) ** (1 / num_steps)
    lrs, losses = [], []
    lr = lr_start
    best_loss = float('inf')

    for step in range(num_steps):
        acts = forward(params, X)
        loss = cross_entropy(acts[-1], y)
        lrs.append(lr)
        losses.append(loss)
        if loss > best_loss * 4:     # stop if loss explodes
            break
        best_loss = min(best_loss, loss)
        backward_and_update(params, acts, y, lr)
        lr *= mult                    # exponential increase

    # Smooth losses with running average for cleaner signal
    smooth = []
    beta = 0.9
    avg = 0
    for i, l in enumerate(losses):
        avg = beta * avg + (1 - beta) * l
        smooth.append(avg / (1 - beta ** (i + 1)))  # bias correction

    return lrs, losses, smooth

if __name__ == "__main__":
    lrs, raw, smooth = lr_range_test(X, y)

    # Find steepest descent: where d(smooth)/d(log_lr) is most negative
    log_lrs = [np.log10(lr) for lr in lrs]
    gradients = np.gradient(smooth, log_lrs)
    best_idx = np.argmin(gradients)
    print(f"Steepest loss descent at LR = {lrs[best_idx]:.4f}")
    print(f"Suggested max LR: {lrs[best_idx]:.4f}")
    # Suggested max LR: ~0.02-0.05 (depends on random seed)
