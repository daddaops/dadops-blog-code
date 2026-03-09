"""Focal loss and FCOS decoding.

Focal loss handles class imbalance in dense detectors.
FCOS-style per-pixel prediction decoding converts distance maps
to bounding boxes.
"""
import numpy as np


def focal_loss(pred_prob, target, alpha=0.25, gamma=2.0):
    """Focal loss: down-weight easy negatives in dense detection."""
    pred_prob = np.clip(pred_prob, 1e-7, 1 - 1e-7)
    # p_t = p for positive, 1-p for negative
    p_t = np.where(target == 1, pred_prob, 1 - pred_prob)
    alpha_t = np.where(target == 1, alpha, 1 - alpha)

    loss = -alpha_t * (1 - p_t)**gamma * np.log(p_t)
    return loss.mean()

def fcos_decode(center_map, distance_map, stride=8, threshold=0.3):
    """Decode FCOS-style per-pixel predictions into boxes."""
    H, W = center_map.shape
    boxes, scores = [], []

    for y in range(H):
        for x in range(W):
            centerness = center_map[y, x]
            if centerness < threshold:
                continue
            l, t, r, b = distance_map[y, x]
            px = (x + 0.5) * stride  # pixel coordinate
            py = (y + 0.5) * stride

            x_min = px - l
            y_min = py - t
            x_max = px + r
            y_max = py + b

            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(centerness)

    return np.array(boxes) if boxes else np.zeros((0,4)), np.array(scores)


if __name__ == "__main__":
    # Focal loss demo: compare standard CE vs focal on an easy negative
    easy_neg_prob = 0.01  # model gives 1% foreground prob (correctly ID's background)
    ce_loss = -np.log(1 - easy_neg_prob)          # standard CE for this sample
    fl_loss = focal_loss(np.array([easy_neg_prob]), np.array([0]))
    print(f"Easy negative: CE={ce_loss:.4f}, Focal={fl_loss:.8f}")
