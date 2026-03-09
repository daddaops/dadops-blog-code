"""2D dilated convolution from scratch."""
import numpy as np

def dilated_conv2d(image, kernel, dilation_rate=1):
    """2D dilated convolution from scratch."""
    H, W = image.shape
    kh, kw = kernel.shape
    # Effective kernel size with dilation
    eff_kh = kh + (kh - 1) * (dilation_rate - 1)
    eff_kw = kw + (kw - 1) * (dilation_rate - 1)
    pad_h, pad_w = eff_kh // 2, eff_kw // 2

    # Pad input
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros((H, W))

    for y in range(H):
        for x in range(W):
            val = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    py = y + pad_h + ky * dilation_rate - pad_h
                    px = x + pad_w + kx * dilation_rate - pad_w
                    if 0 <= py < padded.shape[0] and 0 <= px < padded.shape[1]:
                        val += padded[py, px] * kernel[ky, kx]
            output[y, x] = val
    return output

# Edge detection kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype=float)

image = np.random.rand(16, 16)
for rate in [1, 2, 4]:
    out = dilated_conv2d(image, kernel, dilation_rate=rate)
    eff_size = 3 + (3 - 1) * (rate - 1)
    print(f"Rate {rate}: effective {eff_size}x{eff_size} receptive field, output {out.shape}")
