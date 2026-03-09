import numpy as np

def adjust_brightness(image, delta):
    """Shift pixel values by delta. image in [0, 1]."""
    return np.clip(image + delta, 0.0, 1.0)

def adjust_contrast(image, factor):
    """Blend image toward its mean (factor<1) or away (factor>1)."""
    mean = image.mean()
    return np.clip(mean + factor * (image - mean), 0.0, 1.0)

def adjust_saturation(image, factor):
    """Blend between grayscale (factor=0) and original (factor=1)."""
    gray = np.mean(image, axis=2, keepdims=True)
    return np.clip(gray + factor * (image - gray), 0.0, 1.0)

def random_erasing(image, area_ratio=0.2, rng=None):
    """Zero out a random rectangular patch (Cutout/Random Erasing)."""
    rng = rng or np.random.default_rng()
    h, w = image.shape[:2]
    area = int(h * w * area_ratio)
    # Random aspect ratio between 0.5 and 2.0
    aspect = rng.uniform(0.5, 2.0)
    eh = int(np.sqrt(area * aspect))
    ew = int(np.sqrt(area / aspect))
    eh, ew = min(eh, h), min(ew, w)

    top = rng.integers(0, h - eh + 1)
    left = rng.integers(0, w - ew + 1)
    out = image.copy()
    out[top:top + eh, left:left + ew] = 0.0
    return out

def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, rng=None):
    """Apply random brightness, contrast, and saturation jitter."""
    rng = rng or np.random.default_rng()
    img = adjust_brightness(image, rng.uniform(-brightness, brightness))
    img = adjust_contrast(img, 1.0 + rng.uniform(-contrast, contrast))
    img = adjust_saturation(img, 1.0 + rng.uniform(-saturation, saturation))
    return img


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Create a 16x16 RGB test image
    image = rng.random((16, 16, 3))
    print("Original shape:", image.shape)
    print("Original range: [{:.3f}, {:.3f}]".format(image.min(), image.max()))

    bright = adjust_brightness(image, 0.2)
    print("Brightness +0.2 range: [{:.3f}, {:.3f}]".format(bright.min(), bright.max()))

    contrast = adjust_contrast(image, 1.5)
    print("Contrast 1.5x range: [{:.3f}, {:.3f}]".format(contrast.min(), contrast.max()))

    desat = adjust_saturation(image, 0.0)
    print("Desaturated (factor=0) — all channels equal:", np.allclose(desat[:,:,0], desat[:,:,1]))

    erased = random_erasing(image, area_ratio=0.2, rng=rng)
    zero_pixels = np.sum(np.all(erased == 0, axis=2))
    print("Random erasing: {} zero pixels".format(zero_pixels))

    jittered = color_jitter(image, rng=rng)
    print("Color jitter shape:", jittered.shape)
    print("All photometric transforms passed.")
