import numpy as np

def coral_loss(source_features, target_features):
    """Compute the CORAL loss between source and target features.

    Measures squared Frobenius norm between covariance matrices,
    normalized by feature dimension.
    """
    d = source_features.shape[1]

    # Center both sets of features
    source_centered = source_features - source_features.mean(axis=0)
    target_centered = target_features - target_features.mean(axis=0)

    # Covariance matrices (using n-1 normalization)
    n_s = len(source_features)
    n_t = len(target_features)
    cov_s = (source_centered.T @ source_centered) / (n_s - 1)
    cov_t = (target_centered.T @ target_centered) / (n_t - 1)

    # Squared Frobenius norm of the difference
    diff = cov_s - cov_t
    loss = np.sum(diff ** 2) / (4 * d * d)
    return loss

# Example: features with different covariance structures
np.random.seed(42)
source = np.random.randn(200, 5) @ np.diag([2, 1, 1, 1, 0.5])
target = np.random.randn(200, 5) @ np.diag([0.5, 1, 1, 1, 2])
print(f"CORAL loss: {coral_loss(source, target):.4f}")

# Same covariance should give near-zero loss
target_same = np.random.randn(200, 5) @ np.diag([2, 1, 1, 1, 0.5])
print(f"Same cov:   {coral_loss(source, target_same):.4f}")
