# Object Detection from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/object-detection-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `iou_giou.py` | IoU and Generalized IoU loss |
| `anchor_matching.py` | Anchor generation and GT matching |
| `nms.py` | Non-Maximum Suppression and Soft-NMS |
| `yolo_head.py` | Simplified YOLO detection head |
| `focal_loss_fcos.py` | Focal loss and FCOS decoding |

## Usage

```bash
python iou_giou.py          # IoU and GIoU loss
python anchor_matching.py   # Anchor generation and matching
python nms.py               # NMS demo
python yolo_head.py         # YOLO detection head
python focal_loss_fcos.py   # Focal loss comparison
```

Dependencies: numpy.
