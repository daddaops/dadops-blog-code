"""IoU and GIoU loss for bounding box regression.

Computes Intersection over Union and Generalized IoU loss
between predicted and ground-truth bounding boxes.
"""
import numpy as np


def compute_iou(boxes_a, boxes_b):
    """IoU between two sets of boxes [x_min, y_min, x_max, y_max]."""
    # Intersection coordinates
    inter_min = np.maximum(boxes_a[:, :2], boxes_b[:, :2])
    inter_max = np.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    inter_wh = np.clip(inter_max - inter_min, 0, None)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union_area = area_a + area_b - inter_area

    return inter_area / np.clip(union_area, 1e-6, None)

def giou_loss(pred, target):
    """Generalized IoU loss for bounding box regression."""
    iou = compute_iou(pred, target)

    # Smallest enclosing box
    enclose_min = np.minimum(pred[:, :2], target[:, :2])
    enclose_max = np.maximum(pred[:, 2:], target[:, 2:])
    enclose_wh = enclose_max - enclose_min
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

    inter_min = np.maximum(pred[:, :2], target[:, :2])
    inter_max = np.minimum(pred[:, 2:], target[:, 2:])
    inter_wh = np.clip(inter_max - inter_min, 0, None)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union_area = area_p + area_t - inter_area

    giou = iou - (enclose_area - union_area) / np.clip(enclose_area, 1e-6, None)
    return 1 - giou  # loss: lower is better


if __name__ == "__main__":
    # Example: predicted vs ground-truth box
    pred = np.array([[50, 50, 200, 200]])
    gt   = np.array([[60, 40, 210, 190]])
    print(f"IoU: {compute_iou(pred, gt)[0]:.3f}")
    print(f"GIoU loss: {giou_loss(pred, gt)[0]:.3f}")
