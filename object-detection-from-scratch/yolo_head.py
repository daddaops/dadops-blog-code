"""Simplified YOLO detection head.

Grid-based single-pass detector that divides the image into cells
and predicts bounding boxes per cell with confidence and class probs.
"""
import numpy as np
from nms import nms


class SimpleYOLOHead:
    """Simplified YOLO: grid-based single-pass detection."""

    def __init__(self, grid_size=7, num_classes=3, img_size=448):
        self.S = grid_size
        self.C = num_classes
        self.img_size = img_size
        self.cell_size = img_size / grid_size

    def decode(self, predictions):
        """Decode raw grid predictions into bounding boxes.

        predictions: [S, S, 5 + C] per cell: (tx, ty, tw, th, conf, class_probs)
        tx, ty are offsets within the cell (0-1)
        tw, th are fractions of image size (0-1)
        """
        S, C = self.S, self.C
        boxes, scores, classes = [], [], []

        for row in range(S):
            for col in range(S):
                cell = predictions[row, col]
                tx, ty, tw, th, conf = cell[:5]
                class_probs = cell[5:5 + C]

                # Decode to image coordinates
                cx = (col + tx) * self.cell_size
                cy = (row + ty) * self.cell_size
                w = tw * self.img_size
                h = th * self.img_size

                x_min = cx - w / 2
                y_min = cy - h / 2
                x_max = cx + w / 2
                y_max = cy + h / 2

                cls = class_probs.argmax()
                score = conf * class_probs[cls]

                if score > 0.1:  # confidence threshold
                    boxes.append([x_min, y_min, x_max, y_max])
                    scores.append(score)
                    classes.append(cls)

        if not boxes:
            return np.zeros((0, 4)), np.array([]), np.array([])

        boxes = np.array(boxes)
        scores = np.array(scores)
        keep = nms(boxes, scores, iou_threshold=0.5)
        return boxes[keep], scores[keep], np.array(classes)[keep]


if __name__ == "__main__":
    # Simulate: 7x7 grid, 3 classes, one object in cell (3, 2)
    preds = np.zeros((7, 7, 8))         # 5 box params + 3 classes
    preds[3, 2] = [0.5, 0.5, 0.3, 0.4, 0.92, 0.05, 0.9, 0.05]

    detector = SimpleYOLOHead(grid_size=7, num_classes=3)
    boxes, scores, classes = detector.decode(preds)
    print(f"Detected {len(boxes)} object(s), class={classes}, conf={scores}")
