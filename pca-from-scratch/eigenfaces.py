"""Eigenfaces demo with synthetic face-like data.

Shows how PCA compresses 256-dimensional face images into a small
number of eigenfaces while preserving most of the variance.
"""
import numpy as np

def pca_via_svd(X, n_components):
    """PCA using SVD."""
    n_samples = X.shape[0]
    mean = X.mean(axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    X_projected = X_centered @ components.T
    explained_var = (S ** 2) / n_samples
    total_var = explained_var.sum()
    explained_var_ratio = explained_var[:n_components] / total_var
    return X_projected, components, explained_var_ratio

def eigenfaces_demo(n_people=10, img_size=16, n_images_per_person=20, seed=42):
    """
    Simulate eigenface decomposition with synthetic face-like data.
    Each 'face' is a flattened image vector with person-specific patterns.
    """
    rng = np.random.RandomState(seed)
    n_pixels = img_size * img_size  # 256 dimensions

    # Generate synthetic 'face templates' — each person has a distinct pattern
    templates = []
    for p in range(n_people):
        template = np.zeros((img_size, img_size))
        # Unique face structure per person (random smooth gradients)
        for freq in range(1, 4):
            phase_x = rng.uniform(0, 2 * np.pi)
            phase_y = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.5, 2.0) / freq
            xs = np.linspace(0, freq * np.pi, img_size)
            ys = np.linspace(0, freq * np.pi, img_size)
            template += amp * np.outer(np.sin(xs + phase_x), np.cos(ys + phase_y))
        templates.append(template.flatten())

    # Generate images: template + noise (lighting, expression variation)
    faces = []
    person_labels = []
    for p in range(n_people):
        for _ in range(n_images_per_person):
            noise = rng.randn(n_pixels) * 0.3
            lighting = rng.uniform(0.7, 1.3)
            face = templates[p] * lighting + noise
            faces.append(face)
            person_labels.append(p)

    X_faces = np.array(faces)
    labels = np.array(person_labels)

    # Apply PCA
    X_proj, components, var_ratios = pca_via_svd(X_faces, n_components=50)

    # How many eigenfaces for 95% variance?
    cumulative = np.cumsum(var_ratios)
    n_for_95 = np.searchsorted(cumulative, 0.95) + 1

    # Reconstruction error with different numbers of components
    mean_face = X_faces.mean(axis=0)
    X_centered = X_faces - mean_face

    print(f"Face dataset: {X_faces.shape[0]} images, {n_pixels} pixels each")
    print(f"Top 5 eigenface variance ratios: {var_ratios[:5].round(4)}")
    print(f"Components for 95% variance: {n_for_95} (of {n_pixels} pixels)")
    print(f"\nReconstruction error (MSE) by number of eigenfaces:")
    for k in [5, 10, 20, 50]:
        proj_k = X_centered @ components[:k].T
        reconstructed = proj_k @ components[:k] + mean_face
        mse = np.mean((X_faces - reconstructed) ** 2)
        print(f"  {k:3d} eigenfaces: MSE = {mse:.4f}")

    return X_proj, labels


X_proj_faces, face_labels = eigenfaces_demo()
