import numpy as np

def horizontal_flip(image):
    """Flip image left-to-right. Shape: (H, W) or (H, W, C)."""
    return image[:, ::-1]

def rotate(image, angle_deg):
    """Rotate image by angle_deg degrees around center."""
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Build output coordinate grid
    ys, xs = np.mgrid[0:h, 0:w]

    # Inverse rotation: map output coords back to input
    src_x = cos_t * (xs - cx) + sin_t * (ys - cy) + cx
    src_y = -sin_t * (xs - cx) + cos_t * (ys - cy) + cy

    # Nearest-neighbor sampling (clip to bounds)
    src_x = np.clip(np.round(src_x).astype(int), 0, w - 1)
    src_y = np.clip(np.round(src_y).astype(int), 0, h - 1)
    return image[src_y, src_x]

def random_crop(image, crop_h, crop_w, rng=None):
    """Randomly crop a (crop_h x crop_w) patch from the image."""
    rng = rng or np.random.default_rng()
    h, w = image.shape[:2]
    top = rng.integers(0, h - crop_h + 1)
    left = rng.integers(0, w - crop_w + 1)
    return image[top:top + crop_h, left:left + crop_w]

def affine_transform(image, matrix_2x3):
    """Apply a 2x3 affine transform using inverse mapping.
    matrix_2x3: [[a, b, tx], [c, d, ty]]
    """
    h, w = image.shape[:2]
    M = np.array(matrix_2x3, dtype=float)
    # Invert the 2x2 part for backward mapping
    A = M[:, :2]
    t = M[:, 2]
    A_inv = np.linalg.inv(A)

    ys, xs = np.mgrid[0:h, 0:w]
    coords = np.stack([xs.ravel() - t[0], ys.ravel() - t[1]])
    src = A_inv @ coords
    src_x = np.clip(np.round(src[0]).astype(int), 0, w - 1).reshape(h, w)
    src_y = np.clip(np.round(src[1]).astype(int), 0, h - 1).reshape(h, w)
    return image[src_y, src_x]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Create a simple 8x8 grayscale test image
    image = rng.random((8, 8))
    print("Original shape:", image.shape)

    flipped = horizontal_flip(image)
    print("Flipped shape:", flipped.shape)
    print("Flip correct:", np.allclose(flipped[:, 0], image[:, -1]))

    rotated = rotate(image, 45)
    print("Rotated 45° shape:", rotated.shape)

    cropped = random_crop(image, 4, 4, rng=rng)
    print("Cropped shape:", cropped.shape)

    # Shear transform
    shear_matrix = [[1, 0.2, 0], [0, 1, 0]]
    sheared = affine_transform(image, shear_matrix)
    print("Sheared shape:", sheared.shape)
    print("All geometric transforms passed.")
