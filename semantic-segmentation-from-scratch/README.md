# Semantic Segmentation from Scratch

Verified, runnable code from the DadOps blog post:
[Semantic Segmentation from Scratch](https://www.dadops.co/blog/semantic-segmentation-from-scratch/)

## Scripts

- **fcn_segment.py** — Minimal FCN pipeline: 1×1 conv + nearest-neighbor upsampling from a 7×7 feature map to 224×224
- **unet_forward.py** — Simplified U-Net with 3-level encoder-decoder and skip connections (concatenation)
- **dilated_conv2d.py** — Dilated convolution from scratch at rates 1, 2, 4 showing expanding receptive fields
- **loss_functions.py** — Pixel-wise cross-entropy, Dice loss, and mIoU metric with class imbalance demo

## Usage

```bash
pip install -r requirements.txt
python3 fcn_segment.py
python3 unet_forward.py
python3 dilated_conv2d.py
python3 loss_functions.py
```
