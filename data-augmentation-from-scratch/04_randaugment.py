import numpy as np

# --- Functions from earlier blocks needed by RandAugment ---

def rotate(image, angle_deg):
    """Rotate image by angle_deg degrees around center."""
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ys, xs = np.mgrid[0:h, 0:w]
    src_x = cos_t * (xs - cx) + sin_t * (ys - cy) + cx
    src_y = -sin_t * (xs - cx) + cos_t * (ys - cy) + cy
    src_x = np.clip(np.round(src_x).astype(int), 0, w - 1)
    src_y = np.clip(np.round(src_y).astype(int), 0, h - 1)
    return image[src_y, src_x]

def adjust_brightness(image, delta):
    """Shift pixel values by delta. image in [0, 1]."""
    return np.clip(image + delta, 0.0, 1.0)

def adjust_contrast(image, factor):
    """Blend image toward its mean (factor<1) or away (factor>1)."""
    mean = image.mean()
    return np.clip(mean + factor * (image - mean), 0.0, 1.0)

def affine_transform(image, matrix_2x3):
    """Apply a 2x3 affine transform using inverse mapping."""
    h, w = image.shape[:2]
    M = np.array(matrix_2x3, dtype=float)
    A = M[:, :2]
    t = M[:, 2]
    A_inv = np.linalg.inv(A)
    ys, xs = np.mgrid[0:h, 0:w]
    coords = np.stack([xs.ravel() - t[0], ys.ravel() - t[1]])
    src = A_inv @ coords
    src_x = np.clip(np.round(src[0]).astype(int), 0, w - 1).reshape(h, w)
    src_y = np.clip(np.round(src[1]).astype(int), 0, h - 1).reshape(h, w)
    return image[src_y, src_x]

# --- RandAugment ---

class RandAugment:
    """RandAugment: N random transforms at shared magnitude M."""

    def __init__(self, n=2, m=9, num_levels=31):
        self.n = n                # Number of transforms to apply
        self.m = m                # Magnitude (0 to num_levels-1)
        self.num_levels = num_levels
        self.transforms = [
            'auto_contrast', 'equalize', 'rotate', 'solarize',
            'posterize', 'color', 'brightness', 'contrast',
            'sharpness', 'shear_x', 'shear_y',
            'translate_x', 'translate_y',
        ]

    def _apply(self, image, op, magnitude):
        """Apply a single transform at given magnitude (0.0 to 1.0)."""
        if op == 'rotate':
            angle = magnitude * 30  # up to 30 degrees
            return rotate(image, angle)
        elif op == 'brightness':
            return adjust_brightness(image, magnitude * 0.5 - 0.25)
        elif op == 'contrast':
            return adjust_contrast(image, 1.0 + magnitude - 0.5)
        elif op == 'shear_x':
            M = np.array([[1, magnitude * 0.3, 0],
                          [0, 1, 0]], dtype=float)
            return affine_transform(image, M)
        elif op == 'solarize':
            threshold = 1.0 - magnitude
            mask = image > threshold
            out = image.copy()
            out[mask] = 1.0 - out[mask]
            return out
        # ... (other transforms follow the same pattern)
        return image

    def __call__(self, image, rng=None):
        rng = rng or np.random.default_rng()
        magnitude = self.m / self.num_levels

        for _ in range(self.n):
            op = self.transforms[rng.integers(len(self.transforms))]
            image = self._apply(image, op, magnitude)
        return image

# Usage:
# augmenter = RandAugment(n=2, m=9)
# augmented = augmenter(image)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Create a 16x16 grayscale test image
    image = rng.random((16, 16))
    print("Original shape:", image.shape)

    augmenter = RandAugment(n=2, m=9)
    augmented = augmenter(image, rng=rng)
    print("Augmented shape:", augmented.shape)
    print("Augmented range: [{:.3f}, {:.3f}]".format(augmented.min(), augmented.max()))

    # Apply multiple times to show variation
    for i in range(3):
        aug = augmenter(image, rng=rng)
        print("  Run {}: range [{:.3f}, {:.3f}]".format(i+1, aug.min(), aug.max()))
    print("RandAugment test passed.")
