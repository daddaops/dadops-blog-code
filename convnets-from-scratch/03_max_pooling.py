"""Max pooling layer from scratch."""
import numpy as np

def max_pool2d(x, pool_size=2):
    """Max pooling over spatial dimensions.
    x:       (C, H, W)
    returns: (C, H//pool_size, W//pool_size)
    """
    c, h, w = x.shape
    out_h = h // pool_size
    out_w = w // pool_size
    output = np.zeros((c, out_h, out_w))

    for ch in range(c):
        for i in range(out_h):
            for j in range(out_w):
                si, sj = i * pool_size, j * pool_size
                window = x[ch, si:si+pool_size, sj:sj+pool_size]
                output[ch, i, j] = np.max(window)

    return output

if __name__ == "__main__":
    # 8 feature maps, 14x14 each
    features = np.random.randn(8, 14, 14)
    pooled = max_pool2d(features)
    print("Before pooling:", features.shape)  # (8, 14, 14)
    print("After pooling:", pooled.shape)     # (8, 7, 7) -- spatial dims halved
