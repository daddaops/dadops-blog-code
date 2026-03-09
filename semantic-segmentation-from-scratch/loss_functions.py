"""Loss functions for pixel-level segmentation."""
import numpy as np

def pixel_cross_entropy(pred_logits, target, class_weights=None):
    """Pixel-wise cross-entropy loss for segmentation."""
    H, W, C = pred_logits.shape
    # Softmax per pixel
    exp_logits = np.exp(pred_logits - pred_logits.max(axis=2, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=2, keepdims=True)

    loss = 0.0
    for y in range(H):
        for x in range(W):
            c = target[y, x]
            p = np.clip(probs[y, x, c], 1e-7, 1.0)
            w = class_weights[c] if class_weights is not None else 1.0
            loss -= w * np.log(p)
    return loss / (H * W)

def dice_loss(pred_probs, target_mask):
    """Dice loss for binary segmentation."""
    pred = pred_probs.flatten()
    target = target_mask.flatten().astype(float)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-7)
    return 1.0 - dice

def mean_iou(pred_labels, target_labels, num_classes):
    """Mean Intersection-over-Union metric."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred_labels == c)
        target_c = (target_labels == c)
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious)

# Demo: tiny segmentation with severe imbalance (5% foreground)
H, W = 20, 20
target = np.zeros((H, W), dtype=int)
target[8:12, 8:12] = 1  # small foreground square (16 of 400 = 4%)

# Model A: predicts all background (achieves 96% pixel accuracy!)
pred_all_bg = np.zeros((H, W), dtype=int)
acc_bg = (pred_all_bg == target).mean()
dice_bg = 1.0 - dice_loss(np.zeros((H, W)), target)

# Model B: finds the foreground
pred_good = target.copy()
pred_good[7:13, 7:13] = 1  # slightly oversized prediction
dice_good = 1.0 - dice_loss(pred_good.astype(float), target)

print(f"All-background: accuracy={acc_bg:.0%}, Dice={dice_bg:.3f}")
print(f"Good prediction: accuracy={(pred_good==target).mean():.0%}, Dice={dice_good:.3f}")
