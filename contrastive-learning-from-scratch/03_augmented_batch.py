import numpy as np


def create_augmented_batch(data, batch_size=16, aug_noise=0.3):
    """
    Create a batch of positive pairs through augmentation.
    data: (M, D) array of "images" (2D points as a toy stand-in).
    Returns: (2*batch_size, D) where [2i] and [2i+1] are positive pairs.
    """
    indices = np.random.choice(len(data), size=batch_size, replace=False)
    batch = data[indices]  # (batch_size, D)

    # Two augmented views — in real SimCLR: crop, color jitter, blur, flip
    # Our toy version: add random Gaussian noise
    view1 = batch + np.random.randn(*batch.shape) * aug_noise
    view2 = batch + np.random.randn(*batch.shape) * aug_noise

    # Interleave: [view1_0, view2_0, view1_1, view2_1, ...]
    augmented = np.empty((2 * batch_size, data.shape[1]))
    augmented[0::2] = view1
    augmented[1::2] = view2

    return augmented


if __name__ == "__main__":
    np.random.seed(42)

    # Create toy data: 3 clusters
    centers = np.array([[2, 2], [-2, 2], [0, -2.5]])
    data = np.vstack([c + np.random.randn(20, 2) * 0.3 for c in centers])

    batch = create_augmented_batch(data, batch_size=8, aug_noise=0.3)
    print(f"Data shape:  {data.shape}")
    print(f"Batch shape: {batch.shape}")
    print(f"\nFirst 3 positive pairs (view1 vs view2):")
    for i in range(3):
        v1 = batch[2 * i]
        v2 = batch[2 * i + 1]
        dist = np.linalg.norm(v1 - v2)
        print(f"  Pair {i}: view1={v1.round(2)}, view2={v2.round(2)}, dist={dist:.3f}")
