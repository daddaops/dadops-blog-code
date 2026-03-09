"""Minimal FCN: 1x1 conv + bilinear upsampling."""
import numpy as np

def fcn_segment(feature_map, num_classes, target_h, target_w):
    """Minimal FCN: 1x1 conv + bilinear upsampling."""
    feat_h, feat_w, channels = feature_map.shape

    # 1x1 convolution: project each spatial position to class scores
    weights = np.random.randn(channels, num_classes) * 0.01
    # Reshape to (H*W, C) @ (C, K) -> (H*W, K) -> (H, W, K)
    flat = feature_map.reshape(-1, channels)
    scores = flat @ weights  # [feat_h*feat_w, num_classes]
    coarse_map = scores.reshape(feat_h, feat_w, num_classes)
    coarse_labels = coarse_map.argmax(axis=2)  # [feat_h, feat_w]

    # Nearest-neighbor upsampling to original resolution
    upsampled = np.zeros((target_h, target_w), dtype=int)
    for y in range(target_h):
        for x in range(target_w):
            src_y = y * feat_h / target_h
            src_x = x * feat_w / target_w
            upsampled[y, x] = coarse_labels[
                min(int(src_y), feat_h - 1),
                min(int(src_x), feat_w - 1)
            ]
    return coarse_labels, upsampled

# Simulate: 7x7 feature map from a backbone, 4 classes
feat = np.random.randn(7, 7, 512)
coarse, full = fcn_segment(feat, num_classes=4, target_h=224, target_w=224)
print(f"Coarse map: {coarse.shape}")   # (7, 7) — very blocky
print(f"Full map:   {full.shape}")     # (224, 224) — upsampled but blurry
print(f"Each coarse pixel covers {224//7}x{224//7} = {(224//7)**2} output pixels")
