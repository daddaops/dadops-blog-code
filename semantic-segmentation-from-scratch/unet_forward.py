"""Simplified U-Net: 3-level encoder-decoder with skip connections."""
import numpy as np

def conv_block(x, out_channels):
    """Simplified conv block: linear projection (simulates conv + ReLU)."""
    in_channels = x.shape[-1]
    W = np.random.randn(in_channels, out_channels) * 0.01
    h, w, c = x.shape
    return np.maximum(0, x.reshape(-1, c) @ W).reshape(h, w, out_channels)

def downsample(x):
    """Stride-2 spatial subsampling (simulates pooling)."""
    h, w, c = x.shape
    return x[:h//2*2:2, :w//2*2:2, :]  # simple stride-2 subsampling

def upsample(x, target_h, target_w):
    """Nearest-neighbor 2x upsampling."""
    return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)[:target_h, :target_w, :]

def unet_forward(image, num_classes=4):
    """Simplified U-Net: 3-level encoder-decoder with skip connections."""
    # Encoder (downsampling path)
    e1 = conv_block(image, 64)           # Level 1: full resolution
    e2 = conv_block(downsample(e1), 128) # Level 2: 1/2 resolution
    e3 = conv_block(downsample(e2), 256) # Level 3: 1/4 resolution
    bottleneck = conv_block(downsample(e3), 512)  # 1/8 resolution

    # Decoder (upsampling path with skip connections)
    d3_up = upsample(bottleneck, e3.shape[0], e3.shape[1])
    d3 = conv_block(np.concatenate([d3_up, e3], axis=2), 256)  # skip!

    d2_up = upsample(d3, e2.shape[0], e2.shape[1])
    d2 = conv_block(np.concatenate([d2_up, e2], axis=2), 128)  # skip!

    d1_up = upsample(d2, e1.shape[0], e1.shape[1])
    d1 = conv_block(np.concatenate([d1_up, e1], axis=2), 64)   # skip!

    # Final 1x1 conv for pixel classification
    W_out = np.random.randn(64, num_classes) * 0.01
    h, w, c = d1.shape
    logits = d1.reshape(-1, c) @ W_out
    return logits.reshape(h, w, num_classes).argmax(axis=2)

# Example: 64x64 "image" with 3 channels
image = np.random.randn(64, 64, 3)
seg_map = unet_forward(image, num_classes=4)
print(f"Input: {image.shape} -> Segmentation: {seg_map.shape}")
# Input: (64, 64, 3) -> Segmentation: (64, 64) — full resolution preserved!
