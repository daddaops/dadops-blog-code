import numpy as np

def group_conv_c4(image, kernel):
    """C4 group convolution: convolve with 4 rotated copies of kernel."""
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded = np.pad(image, pad, mode='wrap')
    outputs = []
    for r in range(4):  # 0, 90, 180, 270 degrees
        rotated_kernel = np.rot90(kernel, k=r)
        out = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                patch = padded[i:i+kh, j:j+kw]
                out[i, j] = np.sum(patch * rotated_kernel)
        outputs.append(out)
    return np.stack(outputs)  # shape: (4, h, w)

# Horizontal edge detector
kernel = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=float)

# Simple test image with a horizontal bar
image = np.zeros((8, 8))
image[3:5, 1:7] = 1.0

# Group convolution: 4 orientation channels
result = group_conv_c4(image, kernel)
print(f"Output shape: {result.shape}")  # (4, 8, 8)

# Rotation 0: strong response (horizontal kernel on horizontal bar)
# Rotation 1: weak response (vertical kernel on horizontal bar)
print(f"Channel 0 (0 deg) max:   {result[0].max():.1f}")   # strong
print(f"Channel 1 (90 deg) max:  {result[1].max():.1f}")    # weak
print(f"Channel 2 (180 deg) max: {np.abs(result[2]).max():.1f}")  # strong
print(f"Channel 3 (270 deg) max: {result[3].max():.1f}")    # weak
# Rotate input 90 deg --> output channels cycle by 1 position
