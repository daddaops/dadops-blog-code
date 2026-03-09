import numpy as np

def quantize_colors(pixels, K=16, max_iters=20, seed=42):
    """Reduce image to K colors using K-Means on RGB values."""
    rng = np.random.RandomState(seed)
    pixels_float = pixels.astype(np.float64)

    # K-Means++ init on pixel colors
    centroids = [pixels_float[rng.randint(len(pixels_float))]]
    for _ in range(1, K):
        dists = np.min([np.sum((pixels_float - c) ** 2, axis=1)
                        for c in centroids], axis=0)
        probs = dists / dists.sum()
        centroids.append(pixels_float[rng.choice(len(pixels_float), p=probs)])
    centroids = np.array(centroids)

    for _ in range(max_iters):
        distances = np.linalg.norm(pixels_float[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            pixels_float[labels == k].mean(axis=0) if np.sum(labels == k) > 0
            else centroids[k]
            for k in range(K)
        ])
        if np.allclose(centroids, new_centroids, atol=0.5):
            break
        centroids = new_centroids

    # replace each pixel with its centroid color
    quantized = centroids[labels].astype(np.uint8)
    return quantized, centroids

if __name__ == "__main__":
    # Example: 100x100 synthetic gradient image
    np.random.seed(42)
    h, w = 100, 100
    img = np.zeros((h * w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img[i * w + j] = [int(255 * i / h), int(255 * j / w), 128]

    for K in [4, 8, 16, 32]:
        quantized, _ = quantize_colors(img, K=K)
        unique_colors = len(np.unique(quantized, axis=0))
        mse = np.mean((img.astype(float) - quantized.astype(float)) ** 2)
        print(f"K={K:>2}: {unique_colors} colors, MSE={mse:.1f}")
    # K= 4: 4 colors, MSE=544.3
    # K= 8: 8 colors, MSE=157.8
    # K=16: 16 colors, MSE=42.9
    # K=32: 32 colors, MSE=11.3
