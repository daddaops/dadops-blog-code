"""Diagnostic function for detecting overfitting."""
import numpy as np

def diagnose_training(train_losses, val_losses, weights_per_epoch):
    """Analyze training curves and weight statistics to diagnose overfitting."""
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    best_val = min(val_losses)
    best_epoch = val_losses.index(best_val)
    gap = final_val - final_train

    # Weight magnitude trend
    early_mag = np.mean([np.mean(np.abs(w)) for w in weights_per_epoch[0]])
    late_mag = np.mean([np.mean(np.abs(w)) for w in weights_per_epoch[-1]])
    mag_growth = late_mag / max(early_mag, 1e-10)

    print("=== Training Diagnosis ===")
    print(f"Train loss: {final_train:.4f}  |  Val loss: {final_val:.4f}")
    print(f"Gap: {gap:.4f}  |  Best val at epoch {best_epoch}")
    print(f"Weight magnitude growth: {mag_growth:.1f}x")

    if gap < 0.15 * final_val and final_train > 0.3:
        print("DIAGNOSIS: Underfitting")
        print("  → Increase model capacity (more layers/units)")
        print("  → Train longer, reduce regularization")
        print("  → Check learning rate (may be too low)")
    elif gap > 0.5 * final_train and mag_growth > 3.0:
        print("DIAGNOSIS: Severe overfitting")
        print("  → Add L2 regularization (try λ=0.01)")
        print("  → Add dropout (try p=0.3)")
        print("  → Use early stopping (patience=10-20)")
        print("  → Get more data or use augmentation")
    elif gap > 0.2 * final_train:
        print("DIAGNOSIS: Mild overfitting")
        print("  → Early stopping should suffice")
        print("  → Consider light L2 (λ=0.001)")
    else:
        print("DIAGNOSIS: Healthy training")
        print("  → Train/val gap is small — no action needed")

# Example usage (no output expected — this is a utility function)
print("diagnose_training() defined successfully")
