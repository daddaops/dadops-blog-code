"""Basic 2D convolution from scratch with Sobel edge detection demo."""
import numpy as np

def conv2d(image, kernel):
    """Convolve a 2D image with a 2D kernel (no padding, stride=1)."""
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(patch * kernel)

    return output

if __name__ == "__main__":
    # Sobel edge detector -- finds vertical edges
    sobel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)

    # Test on a 7x7 image with a vertical line in the middle
    image = np.zeros((7, 7), dtype=np.float32)
    image[:, 3] = 1.0  # vertical line at column 3

    edges = conv2d(image, sobel_x)
    print("Input shape:", image.shape)   # (7, 7)
    print("Output shape:", edges.shape)  # (5, 5) -- shrank by kernel_size - 1
    print("Edge map:\n", edges)          # Strong response at the line's borders
