"""Non-Maximum Suppression (NMS) and Soft-NMS.

Standard greedy NMS and Soft-NMS with Gaussian decay to suppress
overlapping detections while preserving nearby distinct objects.
"""
import numpy as np
from iou_giou import compute_iou


def nms(boxes, scores, iou_threshold=0.5):
    """Standard greedy Non-Maximum Suppression."""
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        remaining = order[1:]
        ious = compute_iou(
            np.tile(boxes[i], (len(remaining), 1)),
            boxes[remaining]
        )
        order = remaining[ious < iou_threshold]

    return np.array(keep)

def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.01):
    """Soft-NMS: decay scores instead of hard removal."""
    n = len(boxes)
    indices = np.arange(n)
    out_scores = scores.copy()

    for i in range(n):
        max_idx = out_scores[indices[i:]].argmax() + i
        # Swap current position with max
        indices[i], indices[max_idx] = indices[max_idx], indices[i]

        best = indices[i]
        remaining = indices[i+1:]
        if len(remaining) == 0:
            break
        ious = compute_iou(
            np.tile(boxes[best], (len(remaining), 1)),
            boxes[remaining]
        )
        # Gaussian decay
        out_scores[remaining] *= np.exp(-ious**2 / sigma)

    keep = indices[out_scores[indices] > score_threshold]
    return keep


if __name__ == "__main__":
    # Example: 5 overlapping boxes around one object
    boxes = np.array([[100,100,200,200],[105,98,205,198],
                      [110,102,210,202],[300,300,400,400],[305,298,405,398]])
    scores = np.array([0.9, 0.85, 0.7, 0.95, 0.6])
    print(f"Before NMS: {len(boxes)} boxes")
    print(f"After NMS:  {len(nms(boxes, scores))} boxes")
