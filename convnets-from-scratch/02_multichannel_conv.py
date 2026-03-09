"""Multi-channel convolution with stride and padding support."""
import numpy as np

def conv2d_multi(x, kernels, biases, stride=1, padding=0):
    """Multi-channel convolution.
    x:       (C_in, H, W)
    kernels: (C_out, C_in, kH, kW)
    biases:  (C_out,)
    returns: (C_out, H_out, W_out)
    """
    c_out, c_in, k_h, k_w = kernels.shape
    _, h, w = x.shape

    # Apply zero padding
    if padding > 0:
        x = np.pad(x, ((0,0), (padding,padding), (padding,padding)))
        _, h, w = x.shape

    out_h = (h - k_h) // stride + 1
    out_w = (w - k_w) // stride + 1
    output = np.zeros((c_out, out_h, out_w))

    for f in range(c_out):            # for each output filter
        for i in range(out_h):
            for j in range(out_w):
                si, sj = i * stride, j * stride
                patch = x[:, si:si+k_h, sj:sj+k_w]   # (C_in, kH, kW)
                output[f, i, j] = np.sum(patch * kernels[f]) + biases[f]

    return output

if __name__ == "__main__":
    # Example: 3 input channels (RGB), 8 output filters, 3x3 kernels
    x = np.random.randn(3, 16, 16)           # RGB image, 16x16
    kernels = np.random.randn(8, 3, 3, 3) * 0.1
    biases = np.zeros(8)

    out = conv2d_multi(x, kernels, biases)
    print("Input:", x.shape)    # (3, 16, 16)
    print("Output:", out.shape)  # (8, 14, 14) -- 8 feature maps, each 14x14
