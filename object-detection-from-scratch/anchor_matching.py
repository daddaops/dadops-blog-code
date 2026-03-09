"""Anchor generation and matching.

Generates anchor boxes at multiple scales/aspect ratios across
a feature map, then matches each anchor to ground-truth objects.
"""
import numpy as np
from iou_giou import compute_iou


def generate_anchors(feat_h, feat_w, stride, scales, ratios):
    """Generate anchors for one FPN level."""
    anchors = []
    for y in range(feat_h):
        for x in range(feat_w):
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            for s in scales:
                for r in ratios:
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)
                    anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    return np.array(anchors)

def match_anchors(anchors, gt_boxes, pos_thresh=0.7, neg_thresh=0.3):
    """Assign each anchor to a ground-truth box or background."""
    n_anchors = len(anchors)
    n_gt = len(gt_boxes)
    labels = -np.ones(n_anchors, dtype=int)  # -1 = ignore
    matched_gt = np.zeros(n_anchors, dtype=int)

    if n_gt == 0:
        labels[:] = 0  # all negative
        return labels, matched_gt

    # IoU matrix: [n_anchors, n_gt]
    ious = np.zeros((n_anchors, n_gt))
    for j in range(n_gt):
        ious[:, j] = compute_iou(anchors, np.tile(gt_boxes[j], (n_anchors, 1)))

    best_gt_per_anchor = ious.argmax(axis=1)
    best_iou_per_anchor = ious.max(axis=1)

    labels[best_iou_per_anchor < neg_thresh] = 0          # negative
    labels[best_iou_per_anchor >= pos_thresh] = 1         # positive
    matched_gt = best_gt_per_anchor

    # Ensure each GT has at least one positive anchor
    best_anchor_per_gt = ious.argmax(axis=0)
    labels[best_anchor_per_gt] = 1
    matched_gt[best_anchor_per_gt] = np.arange(n_gt)

    return labels, matched_gt


if __name__ == "__main__":
    # Example: 4x4 feature map, stride 16, one scale, one ratio
    anchors = generate_anchors(4, 4, stride=16, scales=[32], ratios=[1.0])
    gt_boxes = np.array([[20, 10, 60, 55]])
    labels, matched = match_anchors(anchors, gt_boxes)
    print(f"Anchors: {len(anchors)}, Positive: {(labels==1).sum()}, Negative: {(labels==0).sum()}")
